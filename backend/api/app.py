"""HTTP + WebSocket API. Contract: docs/api-contracts.md (kept in sync with this file by tests/test_api.py).

Run: python -m scripts.lab serve   (binds 127.0.0.1 by default; there is no authentication, keep it local)
"""
from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psutil
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.api import game
from backend.game.engine import DIFFICULTIES, ENGINE_VERSION
from backend.models import registry
from backend.research.hypotheses.library import seed_library
from backend.research.orchestration import controller
from backend.research.reporting import report
from backend.rl.environments.env import ENV_VERSION, N_ACTIONS, OBS_DIM, OBS_NAMES, OBS_VERSION
from backend.rl.evaluation.evaluate import OOD_SCENARIOS, SCENARIOS, THRESHOLDS, evaluate_model
from backend.rl.experimental import VARIANTS
from backend.rl.training import train
from backend.storage import db


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    seed_library()
    train.reconcile()
    controller.reconcile()
    yield


app = FastAPI(title="Flappy RL Research Lab API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


def limits() -> dict:
    return controller.load_profiles()["limits"]


def need(row, what: str):
    if row is None:
        raise HTTPException(404, f"{what} not found")
    return row


def bad(exc: Exception) -> HTTPException:
    return HTTPException(422, str(exc).strip("'\""))


# ------------------------------------------------------------------ system
@app.get("/api/health")
def health():
    return {"ok": True, "api_version": app.version, "engine_version": ENGINE_VERSION, "env_version": ENV_VERSION}


_gpu_cache: tuple[float, dict | None] = (0.0, None)


def gpu_status() -> dict | None:
    global _gpu_cache
    if time.time() - _gpu_cache[0] < 2.0:
        return _gpu_cache[1]
    info = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
            name, util, used, total = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            info = {"name": name, "utilization_percent": float(util), "memory_used_mb": float(used),
                    "memory_total_mb": float(total)}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            info = None
    _gpu_cache = (time.time(), info)
    return info


@app.get("/api/status")
def status():
    """Everything the Overview screen needs in one call."""
    vm = psutil.virtual_memory()
    active = train.active_runs()
    session = controller.active_session()
    exp = session and db.one("SELECT e.*, h.title AS hypothesis_title FROM experiments e JOIN hypotheses h "
                             "ON h.id=e.hypothesis_id WHERE e.session_id=? AND e.status='running' ORDER BY e.id DESC",
                             (session["id"],))
    best = registry.best_verified()
    top = db.one("SELECT MAX(max) AS best_score FROM evals")
    lim = limits()
    return {
        "time": time.time(),
        "research": {"status": session["status"] if session else "idle", "session": session, "active_experiment": exp},
        "training": {"active_runs": [{k: r[k] for k in ("id", "algo", "variant", "seed", "status", "steps",
                                                         "total_steps", "experiment_id", "arm")} for r in active],
                     "max_concurrent": lim["max_concurrent"]},
        "totals": {
            "experiments_completed": db.one("SELECT COUNT(*) AS n FROM experiments WHERE status='completed'")["n"],
            "experiments_failed": db.one("SELECT COUNT(*) AS n FROM experiments WHERE status='failed'")["n"],
            "runs_completed": db.one("SELECT COUNT(*) AS n FROM runs WHERE status='completed'")["n"],
            "training_steps": db.one("SELECT COALESCE(SUM(steps),0) AS n FROM runs")["n"],
            "models": db.one("SELECT COUNT(*) AS n FROM models")["n"],
            "matches": db.one("SELECT COUNT(*) AS n FROM matches")["n"],
            "held_out_test_uses": db.kv_get("test_set_uses", 0),
        },
        "best_verified_model": best and {**{k: best[k] for k in ("id", "algo", "variant", "train_steps", "status")},
                                         "evidence": (db.kv_get("best_verified") or {}).get("evidence")},
        "best_score": top["best_score"],
        "resources": {"cpu_percent": psutil.cpu_percent(), "cpu_count": psutil.cpu_count(),
                      "ram_used_mb": vm.used / 2**20, "ram_total_mb": vm.total / 2**20, "gpu": gpu_status(),
                      "disk_used_mb": controller.disk_usage_mb(), "disk_limit_mb": lim["max_disk_mb"],
                      "device_benchmark": db.kv_get("device_benchmark")},
        "limits": lim,
    }


@app.get("/api/events")
def events(since_id: int = 0, limit: int = Query(100, ge=1, le=500)):
    return db.query("SELECT * FROM events WHERE id>? ORDER BY id DESC LIMIT ?", (since_id, limit))


# ------------------------------------------------------------------ game
@app.get("/api/game/config")
def game_config():
    return {"engine_version": ENGINE_VERSION, "modes": {k: list(v) for k, v in game.MODES.items()},
            "difficulties": {k: v.to_dict() for k, v in DIFFICULTIES.items()}, "trained_on": "normal",
            "observation": {"version": OBS_VERSION, "dim": OBS_DIM, "names": OBS_NAMES},
            "actions": {"0": "do not flap", "1": "flap"}, "n_actions": N_ACTIONS,
            "speed_range": list(game.SPEED_RANGE), "view_width_range": list(game.VIEW_WIDTH_RANGE),
            "default_limit_ticks": game.DEFAULT_LIMIT_TICKS,
            "max_limit_ticks": game.MAX_LIMIT_TICKS}


@app.websocket("/ws/game")
async def ws_game(ws: WebSocket):
    await game.session(ws)


@app.get("/api/matches")
def matches(mode: str | None = None, limit: int = Query(50, ge=1, le=500)):
    sql, params = "SELECT id, mode, seed, difficulty, participants, result, created_at FROM matches", []
    if mode:
        sql, params = sql + " WHERE mode=?", [mode]
    return db.query(sql + " ORDER BY id DESC LIMIT ?", [*params, limit])


@app.get("/api/matches/stats")
def match_stats():
    """Persistent competition statistics: human record and head-to-head tallies."""
    rows = db.query("SELECT mode, difficulty, result FROM matches")
    human = {"games": 0, "best": {}, "vs_ai": {"wins": 0, "losses": 0, "draws": 0}}
    pairs: dict[tuple, dict] = {}
    for m in rows:
        res = m["result"]
        ps = res["players"]
        for p in ps:
            if p["kind"] == "human":
                human["games"] += 1
                human["best"][m["difficulty"]] = max(human["best"].get(m["difficulty"], 0), p["score"])
        if m["mode"] == "human_vs_ai":
            key = "draws" if res["draw"] else ("wins" if ps[res["winner"]]["kind"] == "human" else "losses")
            human["vs_ai"][key] += 1
        if len(ps) == 2:
            a, b = (p["model_id"] or "human" for p in ps)
            flip = a > b
            k = (b, a) if flip else (a, b)
            rec = pairs.setdefault(k, {"a": k[0], "b": k[1], "wins_a": 0, "wins_b": 0, "draws": 0, "n": 0})
            rec["n"] += 1
            if res["draw"]:
                rec["draws"] += 1
            else:
                rec["wins_b" if (res["winner"] == 1) != flip else "wins_a"] += 1
    return {"human": human, "head_to_head": list(pairs.values()), "total": len(rows)}


@app.get("/api/matches/{match_id}")
def match_detail(match_id: int):
    row = need(db.one("SELECT * FROM matches WHERE id=?", (match_id,)), "match")
    try:
        row["replay"] = game.load_replay(match_id)
    except (KeyError, ValueError, OSError) as exc:
        row["replay"], row["replay_error"] = None, str(exc)
    return row


class SeriesIn(BaseModel):
    model_a: str
    model_b: str
    n_seeds: int = Field(5, ge=1, le=25)
    first_seed: int = Field(30_000, ge=0, lt=2**31)  # competition seeds: disjoint from train/validation/test
    difficulty: Literal["easy", "normal", "hard"] = "normal"
    limit_ticks: int = Field(4_600, ge=1, le=game.MAX_LIMIT_TICKS)


@app.post("/api/competitions/series")
def series(body: SeriesIn):
    try:
        return game.run_series(body.model_a, body.model_b,
                               list(range(body.first_seed, body.first_seed + body.n_seeds)),
                               body.difficulty, body.limit_ticks)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'\""))
    except (registry.IncompatibleModel, FileNotFoundError, ValueError) as exc:
        raise bad(exc)


# ------------------------------------------------------------------ models
@app.get("/api/models")
def models(status: str | None = None, algo: str | None = None, include_unlisted: bool = Query(False, alias="all")):
    """The curated list (one best checkpoint per algorithm/variant, plus promoted and still-training models).
    `?all=true` returns every registered checkpoint."""
    return registry.list_models(status, algo, include_unlisted)


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    row = need(registry.get(model_id), "model")
    row["evals"] = db.query("SELECT id, purpose, split, scenario, step, n, max_score, mean, median, max, min, std, "
                            "success, created_at FROM evals WHERE model_id=? ORDER BY id", (model_id,))
    row["checkpoint_exists"] = db.resolve(row["checkpoint_path"]).exists()
    return row


class StatusIn(BaseModel):
    status: Literal["experimental", "evaluated", "archived"]


@app.post("/api/models/{model_id}/status")
def model_status(model_id: str, body: StatusIn):
    row = need(registry.get(model_id), "model")
    if row["status"] in ("promoted", "training"):
        raise HTTPException(409, f"a {row['status']} model cannot be re-labelled manually")
    registry.set_status(model_id, body.status)
    return registry.get(model_id)


class ActIn(BaseModel):
    obs: list[float] = Field(min_length=OBS_DIM, max_length=OBS_DIM)


@app.post("/api/models/{model_id}/act")
def model_act(model_id: str, body: ActIn):
    need(registry.get(model_id), "model")
    try:
        action, diag = registry.load(model_id).act_diag(body.obs)
    except (registry.IncompatibleModel, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc))
    return {"model_id": model_id, "action": action, **diag}


class EvalIn(BaseModel):
    split: Literal["validation", "test"] = "validation"
    scenarios: list[str] = ["default"]
    episodes: int = Field(50, ge=1, le=200)
    acknowledge_test_use: bool = False


@app.post("/api/models/{model_id}/evaluate")
def model_evaluate(model_id: str, body: EvalIn):
    need(registry.get(model_id), "model")
    unknown = set(body.scenarios) - set(SCENARIOS)
    if unknown or not body.scenarios:
        raise HTTPException(422, f"unknown scenarios {sorted(unknown)}; choose from {sorted(SCENARIOS)}")
    if body.split == "test" and not body.acknowledge_test_use:
        raise HTTPException(409, "the test split is held out for confirmatory evaluation; "
                                 "resend with acknowledge_test_use=true (the use is logged)")
    try:
        res = evaluate_model(model_id, body.split, body.scenarios, body.episodes,
                             "confirmatory" if body.split == "test" else "manual")
    except (registry.IncompatibleModel, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc))
    return [{k: v for k, v in r.items() if k != "scores"} for r in res]


# ------------------------------------------------------------------ training
@app.get("/api/algorithms")
def algorithms():
    hyps = {h["variant"]: h for h in db.query("SELECT id, variant, title, area, status, novelty FROM hypotheses")}
    return {"algorithms": [{"id": k, "name": v, "kind": "baseline", "default_hyperparameters": train.HYPER[k],
                            "supports_variants": k == "xdqn"} for k, v in train.ALGOS.items()],
            "variants": [{"id": v, "hypothesis": hyps.get(v), "is_control": v == "control"} for v in VARIANTS],
            "variant_composition": "join variant ids with '+' (e.g. 'double+nstep') to train a hybrid",
            "thresholds": list(THRESHOLDS), "scenarios": {"in_distribution": ["default"], "ood": OOD_SCENARIOS}}


@app.get("/api/profiles")
def profiles():
    cfg = controller.load_profiles()
    return {"limits": cfg["limits"], "profiles": cfg["profiles"],
            "overridable": {k: {"min": lo, "max": hi} for k, (lo, hi) in controller.OVERRIDABLE.items()}}


class RunIn(BaseModel):
    algo: Literal["dqn", "ppo", "xdqn"]
    variant: str | None = None
    seed: int = Field(0, ge=0, lt=2**31)
    total_steps: int | None = Field(None, ge=1000, le=2_000_000)
    profile: str | None = None
    hyperparameters: dict | None = None
    eval_every: int | None = Field(None, ge=500, le=1_000_000)
    eval_episodes: int | None = Field(None, ge=2, le=200)
    device: Literal["auto", "cpu", "cuda"] = "auto"


def _capacity():
    lim = limits()
    if len(train.active_runs()) >= lim["max_concurrent"]:
        raise HTTPException(409, f"concurrent training limit reached ({lim['max_concurrent']}); "
                                 "stop a run or wait for one to finish")
    used = controller.disk_usage_mb()
    if used > lim["max_disk_mb"]:
        raise HTTPException(409, f"disk budget exceeded ({used:.0f} MB of {lim['max_disk_mb']} MB)")


@app.post("/api/training/runs", status_code=201)
def start_run(body: RunIn):
    _capacity()
    try:
        b = controller.budgets_for(body.profile or "experimental_screening")
        rid = train.create_run(body.algo, body.variant, body.seed, body.total_steps or b["steps"],
                               hyper=body.hyperparameters, profile=body.profile or "manual", arm="manual",
                               eval_every=body.eval_every or b["eval_every"],
                               eval_episodes=body.eval_episodes or b["eval_episodes"],
                               checkpoint_every=b["checkpoint_every"], limits=controller.run_limits(b),
                               device=body.device)
    except (ValueError, TypeError) as exc:
        raise bad(exc)
    train.launch(rid)
    return db.one("SELECT * FROM runs WHERE id=?", (rid,))


@app.get("/api/training/runs")
def list_runs(status: str | None = None, experiment_id: int | None = None, limit: int = Query(100, ge=1, le=1000)):
    sql, params = "SELECT * FROM runs WHERE 1=1", []
    if status:
        sql, params = sql + " AND status=?", params + [status]
    if experiment_id is not None:
        sql, params = sql + " AND experiment_id=?", params + [experiment_id]
    return db.query(sql + " ORDER BY id DESC LIMIT ?", [*params, limit])


@app.get("/api/training/runs/{run_id}")
def run_detail(run_id: int):
    run = need(db.one("SELECT * FROM runs WHERE id=?", (run_id,)), "run")
    run["summary"] = report.run_summary(run)
    run["metric_keys"] = [r["key"] for r in db.query("SELECT DISTINCT key FROM metrics WHERE run_id=?", (run_id,))]
    return run


@app.get("/api/training/runs/{run_id}/metrics")
def run_metrics(run_id: int, keys: str | None = None, since_step: int = -1):
    """Time series as {key: [[step, value], ...]}. Poll with since_step = last step received."""
    need(db.one("SELECT id FROM runs WHERE id=?", (run_id,)), "run")
    sql, params = "SELECT step, key, value FROM metrics WHERE run_id=? AND step>?", [run_id, since_step]
    if keys:
        ks = [k for k in keys.split(",") if k]
        sql, params = sql + f" AND key IN ({','.join('?' * len(ks))})", params + ks
    out: dict[str, list] = {}
    for r in db.query(sql + " ORDER BY step", params):
        out.setdefault(r["key"], []).append([r["step"], r["value"]])
    return out


@app.get("/api/training/runs/{run_id}/evals")
def run_evals(run_id: int):
    return db.query("SELECT id, purpose, split, scenario, step, n, max_score, mean, median, max, min, std, success, "
                    "created_at FROM evals WHERE run_id=? ORDER BY step, id", (run_id,))


@app.get("/api/training/runs/{run_id}/log")
def run_log(run_id: int, tail: int = Query(100, ge=1, le=2000)):
    path = db.experiments_dir() / "logs" / f"run_{run_id:05d}.log"
    lines = path.read_text(errors="replace").splitlines()[-tail:] if path.exists() else []
    return {"run_id": run_id, "lines": lines}


@app.post("/api/training/runs/{run_id}/stop")
def stop_run(run_id: int):
    run = need(db.one("SELECT * FROM runs WHERE id=?", (run_id,)), "run")
    if run["status"] not in ("starting", "running"):
        raise HTTPException(409, f"run is {run['status']}, not running")
    train.request_stop(run_id)  # the trainer checkpoints and exits at its next log interval (<= 1000 steps)
    return {"run_id": run_id, "stop_requested": True}


@app.post("/api/training/runs/{run_id}/resume")
def resume_run(run_id: int):
    run = need(db.one("SELECT * FROM runs WHERE id=?", (run_id,)), "run")
    if run["status"] not in ("stopped", "interrupted", "limit"):
        raise HTTPException(409, f"only stopped, interrupted or limit-hit runs can resume (run is {run['status']})")
    if not run["checkpoint_dir"] or not (db.resolve(run["checkpoint_dir"]) / "state.json").exists():
        raise HTTPException(409, "no checkpoint to resume from")
    _capacity()
    train.launch(run_id, resume=True)
    return db.one("SELECT * FROM runs WHERE id=?", (run_id,))


# ------------------------------------------------------------------ comparison
@app.get("/api/benchmarks")
def benchmarks():
    return {"benchmarks": report.benchmarks(), "metrics": report.LEADERBOARD_METRICS}


def _benchmark(total_steps: int | None, env_version: str | None) -> tuple[int, str]:
    if total_steps and env_version:
        return total_steps, env_version
    bms = report.benchmarks()
    if not bms:
        raise HTTPException(404, "no completed runs yet")
    pick = next((b for b in bms if (not total_steps or b["total_steps"] == total_steps)
                 and (not env_version or b["env_version"] == env_version)), None)
    if pick is None:
        raise HTTPException(404, "no completed runs for that benchmark")
    return pick["total_steps"], pick["env_version"]


@app.get("/api/compare")
def compare(total_steps: int | None = None, env_version: str | None = None, groups: str | None = None):
    """Group summaries (mean curve with CI band, final stats, sample efficiency, cost, generalization) for ONE
    benchmark setting. Defaults to the benchmark with most completed runs."""
    steps, env = _benchmark(total_steps, env_version)
    wanted = set(groups.split(",")) if groups else None
    out = []
    for g in report.groups(steps, env):
        if wanted and g["group_key"] not in wanted:
            continue
        g["generalization"] = report.generalization(g["run_ids"])
        out.append(g)
    return {"benchmark": {"total_steps": steps, "env_version": env, "validation": "16 fixed episodes per periodic "
                          "evaluation, 50 for the final policy, score cap 200"}, "groups": out}


@app.get("/api/compare/pair")
def compare_pair(a: str, b: str):
    """Statistical comparison of candidate group `a` against group `b` (per-seed values, bootstrap CI)."""
    gs = {g["group_key"]: g for g in report.groups()}
    if a not in gs or b not in gs:
        raise HTTPException(404, "group not found")
    ga, gb = gs[a], gs[b]
    if (ga["total_steps"], ga["env_version"]) != (gb["total_steps"], gb["env_version"]):
        raise HTTPException(409, "groups belong to different benchmarks and cannot be compared")
    return {"a": ga["label"], "b": gb["label"],
            "final_mean": report.compare(ga["final"]["values"], gb["final"]["values"]),
            "auc": report.compare(ga["auc"]["values"], gb["auc"]["values"]),
            "wall_ratio": ga["wall_seconds_mean"] / max(gb["wall_seconds_mean"], 1e-9)}


@app.get("/api/leaderboard")
def leaderboard(metric: str = "final_mean", total_steps: int | None = None, env_version: str | None = None):
    steps, env = _benchmark(total_steps, env_version)
    try:
        rows = report.leaderboard(metric, steps, env)
    except ValueError as exc:
        raise bad(exc)
    return {"metric": metric, "description": report.LEADERBOARD_METRICS[metric],
            "lower_is_better": metric.startswith("steps_to"),
            "benchmark": {"total_steps": steps, "env_version": env}, "rows": rows}


# ------------------------------------------------------------------ research
@app.get("/api/research/hypotheses")
def hypotheses():
    return db.query("SELECT * FROM hypotheses ORDER BY id")


@app.get("/api/research/experiments")
def experiments(session_id: int | None = None, hypothesis_id: str | None = None):
    sql, params = ("SELECT e.*, h.title AS hypothesis_title, h.variant AS variant FROM experiments e "
                   "JOIN hypotheses h ON h.id=e.hypothesis_id WHERE 1=1"), []
    if session_id is not None:
        sql, params = sql + " AND e.session_id=?", params + [session_id]
    if hypothesis_id:
        sql, params = sql + " AND e.hypothesis_id=?", params + [hypothesis_id]
    return db.query(sql + " ORDER BY e.id DESC", params)


@app.get("/api/research/experiments/{exp_id}")
def experiment_detail(exp_id: int):
    exp = need(db.one("SELECT * FROM experiments WHERE id=?", (exp_id,)), "experiment")
    exp["hypothesis"] = db.one("SELECT * FROM hypotheses WHERE id=?", (exp["hypothesis_id"],))
    path = exp.get("report_path")
    exp["report_markdown"] = db.resolve(path).read_text() if path and db.resolve(path).exists() else None
    exp["runs"] = db.query("SELECT id, arm, variant, seed, status, steps, total_steps, wall_seconds, model_id, error "
                           "FROM runs WHERE experiment_id=? ORDER BY id", (exp_id,))
    return exp


@app.get("/api/research/observations")
def observations(profile: str = "experimental_screening"):
    try:
        return controller.observe(controller.budgets_for(profile))
    except ValueError as exc:
        raise bad(exc)


@app.get("/api/research/queue")
def research_queue(profile: str = "experimental_screening"):
    """What the controller would run next under `profile`, in order, with the reason for each."""
    try:
        b = controller.budgets_for(profile)
    except ValueError as exc:
        raise bad(exc)
    return controller.queue(controller.observe(b), b)


@app.get("/api/research/sessions")
def sessions():
    return db.query("SELECT * FROM sessions ORDER BY id DESC")


@app.get("/api/research/sessions/{session_id}")
def session_detail(session_id: int):
    s = need(db.one("SELECT * FROM sessions WHERE id=?", (session_id,)), "session")
    s["experiments"] = experiments(session_id=session_id)
    return s


class SessionIn(BaseModel):
    profile: str = "experimental_screening"
    objective: str = Field("", max_length=2000)
    max_experiments: int | None = None
    steps: int | None = None
    seeds: int | None = None
    session_wall_seconds: int | None = None
    max_concurrent: int | None = None
    max_memory_mb: int | None = None
    max_gpu_mem_mb: int | None = None
    max_disk_mb: int | None = None
    early_stop_patience: int | None = None
    checkpoint_every: int | None = None


@app.post("/api/research/sessions", status_code=201)
def start_session(body: SessionIn):
    if controller.active_session():
        raise HTTPException(409, "a research session is already active; stop it or wait for it to finish")
    try:
        sid = controller.create_session(body.profile, body.objective,
                                        body.model_dump(exclude={"profile", "objective"}))
    except ValueError as exc:
        raise bad(exc)
    controller.launch_session(sid)
    return db.one("SELECT * FROM sessions WHERE id=?", (sid,))


@app.post("/api/research/sessions/{session_id}/{action}")
def control_session(session_id: int, action: Literal["pause", "resume", "stop"]):
    s = need(db.one("SELECT * FROM sessions WHERE id=?", (session_id,)), "session")
    if s["status"] not in ("queued", "running", "paused"):
        raise HTTPException(409, f"session is {s['status']}")
    controller.set_control(session_id, {"pause": "pause", "resume": "run", "stop": "stop"}[action])
    return db.one("SELECT * FROM sessions WHERE id=?", (session_id,))


# ------------------------------------------------------------------ frontend (built by Codex into frontend/dist)
_dist = db.ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
