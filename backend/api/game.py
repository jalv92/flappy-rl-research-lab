"""Live matches over WebSocket. The server owns the simulation (the same engine used for training);
clients render snapshots and send inputs. seed + recorded flap ticks make every match exactly replayable.
Protocol: docs/api-contracts.md.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from dataclasses import replace

from fastapi import WebSocket, WebSocketDisconnect

from backend.game.engine import DIFFICULTIES, ENGINE_VERSION, World
from backend.models import registry
from backend.rl.environments.env import observe
from backend.storage import db

MODES = {"human": ("human",), "ai": ("ai",), "human_vs_ai": ("human", "ai"), "ai_vs_ai": ("ai", "ai")}
DEFAULT_LIMIT_TICKS = 27_000   # 15 simulated minutes: guarantees termination when agents never crash
MAX_LIMIT_TICKS = 108_000
FRAME_DT = 1 / 60
MAX_TICKS_PER_FRAME = 16
SPEED_RANGE = (0.25, 16.0)
VIEW_WIDTH_RANGE = (432.0, 1296.0)  # how much of the world a snapshot shows; a renderer choice, never physics


class Match:
    """Mode-independent core: one shared World plus one action source per player."""

    def __init__(self, mode: str, model_ids: list[str] | None = None, seed: int | None = None,
                 difficulty: str = "normal", limit_ticks: int | None = None, recorded: dict | None = None,
                 view_width: float | None = None):
        if recorded:  # replay of a stored match: actions come from the recording, no model is needed
            self.mode, self.replay_of = recorded["mode"], recorded["match_id"]
            difficulty, seed, limit_ticks = recorded["difficulty"], recorded["seed"], recorded["ticks"]
            self.players = [{**p, "policy": None} for p in recorded["participants"]]
            self._recorded = [set(t) for t in recorded["flap_ticks"]]
        else:
            if mode not in MODES:
                raise ValueError(f"mode must be one of {sorted(MODES)}")
            kinds, ids = MODES[mode], list(model_ids or [])
            if len(ids) != kinds.count("ai"):
                raise ValueError(f"mode {mode!r} needs exactly {kinds.count('ai')} model id(s), got {len(ids)}")
            self.mode, self.replay_of, self._recorded, self.players = mode, None, None, []
            for kind in kinds:
                if kind == "human":
                    self.players.append({"kind": "human", "label": "You", "model_id": None, "policy": None})
                else:
                    policy = registry.load(ids.pop(0))  # raises KeyError / IncompatibleModel / FileNotFoundError
                    row = policy.row
                    self.players.append({"kind": "ai", "label": row["id"], "model_id": row["id"], "policy": policy,
                                         "algo": row["algo"], "variant": row["variant"], "status": row["status"],
                                         "train_steps": row["train_steps"], "diagnostic": policy.diag_kind})
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
        limit = DEFAULT_LIMIT_TICKS if limit_ticks is None else int(limit_ticks)
        if not 1 <= limit <= MAX_LIMIT_TICKS:
            raise ValueError(f"limit_ticks must be within 1..{MAX_LIMIT_TICKS}")
        cfg = DIFFICULTIES[difficulty]
        if view_width is not None:  # wider window onto the same world: more obstacles ahead, identical dynamics
            if not VIEW_WIDTH_RANGE[0] <= float(view_width) <= VIEW_WIDTH_RANGE[1]:
                raise ValueError(f"view_width must be within {VIEW_WIDTH_RANGE[0]:.0f}..{VIEW_WIDTH_RANGE[1]:.0f}")
            cfg = replace(cfg, view_width=float(view_width))
        self.difficulty, self.cfg, self.limit_ticks = difficulty, cfg, limit
        self.seed = secrets.randbelow(2**31) if seed is None else int(seed)
        self.world = World(self.cfg, self.seed, len(self.players))
        self.flap_ticks: list[list[int]] = [[] for _ in self.players]
        self.diag: list[dict | None] = [None] * len(self.players)
        self.flapped = [False] * len(self.players)
        self.has_human = any(p["kind"] == "human" for p in self.players) and not recorded

    @property
    def over(self) -> bool:
        return self.world.done or self.world.tick >= self.limit_ticks

    @property
    def human_alive(self) -> bool:
        """True while a human-controlled bird is still flying (playback speed is locked at 1x for them)."""
        return self.has_human and any(b.alive for p, b in zip(self.players, self.world.birds) if p["kind"] == "human")

    def tick(self, human_flap: bool = False) -> None:
        world, t = self.world, self.world.tick + 1
        actions = []
        for i, (player, bird) in enumerate(zip(self.players, world.birds)):
            self.diag[i] = None
            if not bird.alive:
                actions.append(0)
            elif self._recorded is not None:
                actions.append(int(t in self._recorded[i]))
            elif player["kind"] == "human":
                actions.append(int(human_flap))
            else:
                obs = observe(world, bird)
                action, diag = player["policy"].act_diag(obs)  # the model's real output, nothing synthesised
                self.diag[i] = {"obs": obs, "action": action, **diag}
                actions.append(action)
        world.step(actions)
        for i, a in enumerate(actions):
            self.flapped[i] = bool(a)
            if a:
                self.flap_ticks[i].append(t)

    def public_players(self) -> list[dict]:
        return [{k: v for k, v in p.items() if k != "policy"} for p in self.players]

    def ready(self) -> dict:
        return {"type": "ready", "mode": self.mode, "seed": self.seed, "difficulty": self.difficulty,
                "in_distribution": self.difficulty == "normal", "limit_ticks": self.limit_ticks,
                "replay_of": self.replay_of, "config": self.cfg.to_dict(), "players": self.public_players(),
                "personal_best": human_best(self.difficulty) if self.has_human else None}

    def state(self, phase: str, speed: float) -> dict:
        snap = self.world.snapshot()
        players = []
        for i, b in enumerate(snap["birds"]):
            p = {"y": b["y"], "vy": b["vy"], "alive": b["alive"], "score": b["score"], "distance": b["distance"],
                 "flapped": self.flapped[i], "death": b["death"]}
            if self.diag[i] is not None:
                p["diag"] = self.diag[i]
            players.append(p)
        return {"type": "state", "phase": phase, "tick": snap["tick"], "time": snap["tick"] / self.cfg.tick_rate,
                "speed": speed, "scroll_x": snap["scroll_x"], "obstacles": snap["obstacles"], "players": players}

    def result(self) -> dict:
        birds = self.world.birds
        ticks = [b.ticks for b in birds]
        winner = None if len(birds) < 2 or ticks[0] == ticks[1] else int(ticks[1] > ticks[0])
        return {"mode": self.mode, "seed": self.seed, "difficulty": self.difficulty, "ticks": self.world.tick,
                "limit_reached": not self.world.done, "winner": winner, "draw": len(birds) == 2 and winner is None,
                "players": [{**p, "score": b.score, "distance": self.world.distance(b), "ticks": b.ticks,
                             "flaps": b.flaps, "death": b.death, "died_tick": b.died_tick}
                            for p, b in zip(self.public_players(), birds)]}

    def save(self) -> dict:
        """Persist a finished match (result row + replay file). Replays of stored matches are not re-saved."""
        res = self.result()
        if self.replay_of is not None:
            return {**res, "match_id": self.replay_of, "replay": True}
        if self.has_human:
            human = next(p for p in res["players"] if p["kind"] == "human")
            best = human_best(self.difficulty)
            res["personal_best"], res["new_record"] = max(best, human["score"]), human["score"] > best
        mid = db.insert("matches", mode=self.mode, seed=self.seed, difficulty=self.difficulty,
                        participants=self.public_players(), result=res, created_at=time.time())
        path = db.experiments_dir() / "replays" / f"match_{mid:06d}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"match_id": mid, "engine_version": ENGINE_VERSION, "mode": self.mode,
                                    "difficulty": self.difficulty, "seed": self.seed, "ticks": self.world.tick,
                                    "participants": self.public_players(), "flap_ticks": self.flap_ticks}))
        db.update("matches", mid, replay_path=db.rel(path))
        return {**res, "match_id": mid}


def human_best(difficulty: str) -> int:
    best = 0
    for m in db.query("SELECT result FROM matches WHERE difficulty=? AND mode IN ('human','human_vs_ai')", (difficulty,)):
        best = max([best, *(p["score"] for p in m["result"]["players"] if p["kind"] == "human")])
    return best


def load_replay(match_id: int) -> dict:
    row = db.one("SELECT replay_path FROM matches WHERE id=?", (match_id,))
    if row is None or not row["replay_path"]:
        raise KeyError(f"no stored replay for match {match_id}")
    with open(db.resolve(row["replay_path"])) as f:
        data = json.load(f)
    if data["engine_version"] != ENGINE_VERSION:
        raise ValueError(f"replay recorded with engine {data['engine_version']}, current is {ENGINE_VERSION}")
    return data


def run_series(model_a: str, model_b: str, seeds: list[int], difficulty: str, limit_ticks: int) -> dict:
    """Headless AI-vs-AI over several shared seeds (each seed = one identical obstacle field for both)."""
    matches = []
    for seed in seeds:
        m = Match("ai_vs_ai", [model_a, model_b], seed, difficulty, limit_ticks)
        while not m.over:
            m.tick()
        matches.append(m.save())
    wins = [sum(m["winner"] == i for m in matches) for i in (0, 1)]
    return {"model_a": model_a, "model_b": model_b, "difficulty": difficulty, "limit_ticks": limit_ticks,
            "n": len(matches), "wins_a": wins[0], "wins_b": wins[1], "draws": len(matches) - sum(wins),
            "mean_score_a": sum(m["players"][0]["score"] for m in matches) / len(matches),
            "mean_score_b": sum(m["players"][1]["score"] for m in matches) / len(matches),
            "matches": [{"match_id": m["match_id"], "seed": m["seed"], "winner": m["winner"], "draw": m["draw"],
                         "scores": [p["score"] for p in m["players"]], "ticks": [p["ticks"] for p in m["players"]]}
                        for m in matches]}


_LOCAL_ORIGIN = re.compile(r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?")


def origin_allowed(ws: WebSocket) -> bool:
    """Browsers send cookies-free but unrestricted cross-site WebSocket requests, so any web page could drive
    this server. Accept only same-origin pages, local dev servers, and non-browser clients (no Origin header)."""
    origin = ws.headers.get("origin")
    if origin is None:
        return True
    return bool(_LOCAL_ORIGIN.fullmatch(origin)) or origin.split("://", 1)[-1] == ws.headers.get("host")


async def session(ws: WebSocket) -> None:
    if not origin_allowed(ws):
        await ws.close(code=1008)  # policy violation, before accepting
        return
    await ws.accept()
    inbox: asyncio.Queue = asyncio.Queue()

    async def reader():
        while True:
            try:
                inbox.put_nowait(json.loads(await ws.receive_text()))
            except json.JSONDecodeError:
                inbox.put_nowait({"type": "_bad"})
            except (WebSocketDisconnect, RuntimeError):
                inbox.put_nowait({"type": "_closed"})
                return

    task = asyncio.create_task(reader())
    match: Match | None = None
    params: dict = {}
    phase, speed, flap, credit, last = "idle", 1.0, False, 0.0, time.monotonic()

    async def send_state():
        await ws.send_json(match.state(phase, speed))

    async def begin(new_params: dict):
        nonlocal match, params, phase, speed, flap, credit, last
        view = new_params.get("view_width")
        candidate = (Match(None, recorded=load_replay(int(new_params["match_id"])), view_width=view)
                     if "match_id" in new_params
                     else Match(new_params["mode"], new_params.get("models"), new_params.get("seed"),
                                new_params.get("difficulty", "normal"), new_params.get("limit_ticks"),
                                view_width=view))
        match, params = candidate, new_params
        speed = 1.0 if match.has_human else min(max(float(new_params.get("speed", 1.0)), SPEED_RANGE[0]), SPEED_RANGE[1])
        phase, flap, credit, last = ("ready" if match.has_human else "running"), False, 0.0, time.monotonic()
        await ws.send_json(match.ready())
        await send_state()

    try:
        while True:
            msgs = [await inbox.get()] if phase != "running" else []
            while not inbox.empty():
                msgs.append(inbox.get_nowait())
            for msg in msgs:
                kind = msg.get("type") if isinstance(msg, dict) else None
                try:
                    if kind == "_closed":
                        return
                    elif kind == "start":
                        await begin({k: msg[k] for k in ("mode", "models", "seed", "difficulty", "limit_ticks", "speed",
                                                         "view_width") if k in msg})
                    elif kind == "replay":
                        await begin({k: msg[k] for k in ("match_id", "speed", "view_width") if k in msg})
                    elif match is None:
                        raise ValueError("no game in progress: send a 'start' message first")
                    elif kind == "flap":
                        if match.has_human and phase in ("ready", "running"):
                            flap = True
                            if phase == "ready":  # the world starts on the human's first flap, for every participant
                                phase, credit, last = "running", 1.0, time.monotonic()
                    elif kind == "pause" and phase == "running":
                        phase = "paused"
                        await send_state()
                    elif kind == "resume" and phase == "paused":
                        phase, last = "running", time.monotonic()
                        await send_state()
                    elif kind == "restart":
                        seed = msg.get("seed", match.seed if msg.get("same_seed") else None)
                        await begin({**params, "seed": seed} if "match_id" not in params else params)
                    elif kind == "set_speed":
                        if match.human_alive:
                            raise ValueError("speed is fixed at 1x while a human is playing")
                        speed = min(max(float(msg["speed"]), SPEED_RANGE[0]), SPEED_RANGE[1])
                        await send_state()
                    elif kind == "stop":
                        match, phase = None, "idle"
                        await ws.send_json({"type": "stopped"})
                    elif kind not in ("pause", "resume"):
                        raise ValueError(f"unknown or malformed message type {kind!r}")
                except (KeyError, ValueError, TypeError, FileNotFoundError, registry.IncompatibleModel) as exc:
                    await ws.send_json({"type": "error", "message": str(exc).strip("'\"")})
            if phase != "running":
                continue
            now = time.monotonic()
            credit = min(credit + (now - last) * match.cfg.tick_rate * speed, MAX_TICKS_PER_FRAME)
            last = now
            stepped = False
            while credit >= 1.0 and not match.over:
                match.tick(flap)
                flap, credit, stepped = False, credit - 1.0, True
            if stepped:
                await send_state()
            if match.over:
                phase = "over"
                result = await asyncio.to_thread(match.save)
                await send_state()
                await ws.send_json({"type": "game_over", "result": result})
            else:
                await asyncio.sleep(FRAME_DT)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
