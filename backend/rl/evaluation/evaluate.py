"""Gameplay evaluation: agents are scored on obstacles cleared, never on training reward.

Splits: "validation" (model selection, learning curves) and "test" (held-out, confirmatory only).
Scenarios: "default" is in-distribution; the others are out-of-distribution physics/layout shifts.
"""
from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Callable

import numpy as np

from backend.game.engine import GameConfig, World
from backend.rl.environments.env import TEST_SEED0, VAL_SEED0, observe

_BASE = GameConfig()
SCENARIOS: dict[str, GameConfig] = {
    "default": _BASE,  # in-distribution
    "narrow_gap": replace(_BASE, gap_min=112.0, gap_max=135.0),
    "wide_spacing": replace(_BASE, obstacle_spacing=260.0),
    "tight_spacing": replace(_BASE, obstacle_spacing=180.0),
    "fast": replace(_BASE, speed=170.0),
    "slow": replace(_BASE, speed=115.0),
    "hard_combo": replace(_BASE, gap_min=115.0, gap_max=140.0, speed=165.0, obstacle_spacing=190.0),
}
OOD_SCENARIOS = [s for s in SCENARIOS if s != "default"]
THRESHOLDS = (10, 25, 50, 100)
SPLIT_SEED0 = {"validation": VAL_SEED0, "test": TEST_SEED0}

ActFn = Callable[[np.ndarray], np.ndarray]  # obs [B, OBS_DIM] float32 -> actions [B]


def play(act_fn: ActFn, config: GameConfig, seeds: list[int], max_score: int = 200) -> list[int]:
    """Run one episode per seed in lockstep with batched inference; return final scores (capped at max_score)."""
    worlds = [World(config, s) for s in seeds]
    live = list(range(len(worlds)))
    while live:
        obs = np.array([observe(worlds[i], worlds[i].birds[0]) for i in live], dtype=np.float32)
        actions = act_fn(obs)
        still = []
        for i, a in zip(live, actions):
            w = worlds[i]
            w.step((int(a),))
            if w.birds[0].alive and w.birds[0].score < max_score:
                still.append(i)
        live = still
    return [w.birds[0].score for w in worlds]


def summarize(scores: list[int]) -> dict:
    return {
        "n": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "min": min(scores),
        "std": statistics.pstdev(scores),
        "success": {str(t): sum(s >= t for s in scores) / len(scores) for t in THRESHOLDS},
        "scores": scores,
    }


def evaluate(act_fn: ActFn, scenario: str = "default", split: str = "validation",
             n_episodes: int = 16, max_score: int = 200) -> dict:
    seed0 = SPLIT_SEED0[split]
    seeds = list(range(seed0, seed0 + n_episodes))
    out = summarize(play(act_fn, SCENARIOS[scenario], seeds, max_score))
    out.update(scenario=scenario, split=split, max_score=max_score, seed0=seed0)
    return out


def evaluate_model(model_id: str, split: str = "validation", scenarios=("default",), n_episodes: int = 50,
                   purpose: str = "manual") -> list[dict]:
    """Evaluate a registered model and persist the result. Every use of the held-out test split is logged."""
    import time

    from backend.models import registry
    from backend.rl.environments.env import ENV_VERSION
    from backend.storage import db

    policy = registry.load(model_id)
    row, out = policy.row, []
    for scenario in scenarios:
        res = evaluate(policy.act, scenario, split, n_episodes)
        res["id"] = db.insert(
            "evals", run_id=row["run_id"], model_id=model_id, purpose=purpose, split=split, scenario=scenario,
            step=row["train_steps"], n=res["n"], max_score=res["max_score"], mean=res["mean"], median=res["median"],
            max=res["max"], min=res["min"], std=res["std"], success=res["success"], scores=res["scores"],
            env_version=ENV_VERSION, created_at=time.time())
        out.append(res)
    if split == "test":
        db.kv_set("test_set_uses", db.kv_get("test_set_uses", 0) + 1)
        db.event("info", "evaluation", f"held-out test split used for {model_id} ({purpose}, {len(out)} scenarios)")
    return out
