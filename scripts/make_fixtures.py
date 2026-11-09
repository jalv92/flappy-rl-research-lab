"""Capture real API responses and WebSocket traces into docs/api-fixtures/ for frontend development.

    python -m scripts.make_fixtures

Everything written here comes from the live backend code against the current lab database
(real trained models, real runs). Human input in the traces is scripted and labelled as such.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend.api.app import app
from backend.storage import db

OUT = db.ROOT / "docs" / "api-fixtures"


def dump(name: str, data) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1))
    print(f"{name}.json")


def trace(client, start: dict, human: bool, keep_states: int = 400) -> list[dict]:
    """Message log of one match. States beyond `keep_states`, and beyond 60 after the first crash, are thinned
    to every 10th to bound size (ticks in the kept states are therefore not always consecutive)."""
    log, n, after = [], 0, 0
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json(start)
        log.append({"dir": "client", "msg": start})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "state":
                n += 1
                after += any(not p["alive"] for p in msg["players"])
                if (n <= keep_states and after <= 60) or n % 10 == 0 or msg["phase"] == "over" or 0 < after <= 60:
                    log.append({"dir": "server", "msg": msg})
                if human:
                    me = msg["players"][0]
                    if msg["phase"] == "ready" or (me["alive"] and me["y"] > 285 and me["vy"] >= 0):
                        ws.send_json({"type": "flap"})
                        log.append({"dir": "client", "msg": {"type": "flap"}, "note": "scripted human input"})
            else:
                log.append({"dir": "server", "msg": msg})
                if msg["type"] in ("game_over", "error"):
                    return log


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        get = lambda url, **p: c.get(url, params=p).json()
        models = get("/api/models")
        by_algo = {}
        for m in models:  # best finished model per algorithm, by final validation mean
            if m["status"] in ("evaluated", "promoted", "experimental") and "final_val" in m["metrics"]:
                cur = by_algo.get(m["algo"])
                if cur is None or m["metrics"]["final_val"]["mean"] > cur["metrics"]["final_val"]["mean"]:
                    by_algo[m["algo"]] = m
        ppo, dqn = by_algo["ppo"]["id"], by_algo["dqn"]["id"]
        xdqn = by_algo.get("xdqn", by_algo["dqn"])["id"]
        weakest = min((m for m in models if "final_val" in m["metrics"]), key=lambda m: m["metrics"]["final_val"]["mean"])["id"]

        dump("ws_human", trace(c, {"type": "start", "mode": "human", "seed": 101}, True))
        dump("ws_ai_ppo", trace(c, {"type": "start", "mode": "ai", "models": [ppo], "seed": 102, "speed": 16,
                                    "limit_ticks": 1200}, False, 300))
        dump("ws_ai_dqn", trace(c, {"type": "start", "mode": "ai", "models": [xdqn], "seed": 102, "speed": 16,
                                    "limit_ticks": 1200}, False, 300))
        dump("ws_human_vs_ai", trace(c, {"type": "start", "mode": "human_vs_ai", "models": [ppo], "seed": 103,
                                         "limit_ticks": 900}, True))
        dump("ws_ai_vs_ai", trace(c, {"type": "start", "mode": "ai_vs_ai", "models": [weakest, ppo], "seed": 104,
                                      "speed": 16, "limit_ticks": 1500}, False, 300))
        last = get("/api/matches", limit=1)[0]["id"]
        dump("ws_replay", trace(c, {"type": "replay", "match_id": last, "speed": 16}, False, 60))
        with c.websocket_connect("/ws/game") as ws:
            ws.send_json({"type": "start", "mode": "ai", "models": ["does-not-exist"]})
            dump("ws_error", [{"dir": "server", "msg": ws.receive_json()}])

        dump("health", get("/api/health"))
        dump("game_config", get("/api/game/config"))
        dump("status", get("/api/status"))
        dump("events", get("/api/events"))
        dump("models", models)
        dump("model_detail", get(f"/api/models/{ppo}"))
        dump("model_act", c.post(f"/api/models/{ppo}/act", json={"obs": [0.5, 0.1, 0.4, 0.5, 0.5, 0.1, 0.1, 0.27,
                                                                           0.9, 0.5, 0.27, 0.7]}).json())
        dump("series", c.post("/api/competitions/series", json={"model_a": weakest, "model_b": ppo, "n_seeds": 3,
                                                                "limit_ticks": 1500}).json())
        dump("matches", get("/api/matches", limit=20))
        dump("matches_stats", get("/api/matches/stats"))
        dump("match_detail", get(f"/api/matches/{last}"))
        dump("algorithms", get("/api/algorithms"))
        dump("profiles", get("/api/profiles"))
        runs = get("/api/training/runs", limit=200)
        dump("runs", runs)
        rid = by_algo["ppo"]["run_id"]
        dump("run_detail", get(f"/api/training/runs/{rid}"))
        dump("run_metrics", get(f"/api/training/runs/{rid}/metrics"))
        dump("run_metrics_dqn", get(f"/api/training/runs/{by_algo.get('xdqn', by_algo['dqn'])['run_id']}/metrics"))
        dump("run_evals", get(f"/api/training/runs/{rid}/evals"))
        dump("benchmarks", get("/api/benchmarks"))
        cmp = get("/api/compare")
        dump("compare", cmp)
        keys = [g["group_key"] for g in cmp["groups"]]
        if len(keys) >= 2:
            dump("compare_pair", get("/api/compare/pair", a=keys[0], b=keys[1]))
        for metric in ("final_mean", "steps_to_50", "test_mean"):
            dump(f"leaderboard_{metric}", get("/api/leaderboard", metric=metric))
        dump("hypotheses", get("/api/research/hypotheses"))
        dump("observations", get("/api/research/observations"))
        dump("queue", get("/api/research/queue"))
        exps = get("/api/research/experiments")
        dump("experiments", exps)
        if exps:
            dump("experiment_detail", get(f"/api/research/experiments/{exps[0]['id']}"))
        sessions = get("/api/research/sessions")
        dump("sessions", sessions)
        if sessions:
            dump("session_detail", get(f"/api/research/sessions/{sessions[0]['id']}"))
        dump("error_404", c.get("/api/models/does-not-exist").json())
        dump("error_409", c.post(f"/api/models/{ppo}/evaluate", json={"split": "test"}).json())
        dump("error_422", c.post("/api/training/runs", json={"algo": "xdqn", "variant": "nope"}).json())


if __name__ == "__main__":
    main()
