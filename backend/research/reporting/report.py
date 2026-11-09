"""Aggregation and statistics over stored runs. Read-only; shared by the API and the research controller.

A *group* is a set of runs that differ only by training seed (same algo, variant, step budget,
hyperparameters, reward, environment version and variant source). Only groups are compared.
"""
from __future__ import annotations

import numpy as np

from backend.rl.evaluation.evaluate import OOD_SCENARIOS, THRESHOLDS
from backend.storage import db


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> list[float]:
    """Percentile bootstrap CI of the mean over independent training seeds."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return [float(v.mean()), float(v.mean())] if len(v) else [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def compare(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Candidate `a` vs control `b` (per-seed scalars). Positive diff = candidate better."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    dbs = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    diff = da - dbs
    wins = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(a.mean() - b.mean()),
            "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "rel_diff": float((a.mean() - b.mean()) / max(abs(b.mean()), 1e-9)),
            "prob_improvement": float(wins), "n_a": len(a), "n_b": len(b)}


def run_summary(run: dict) -> dict:
    evals = db.query("SELECT step, purpose, mean FROM evals WHERE run_id=? AND split='validation' "
                     "AND scenario='default' ORDER BY step, id", (run["id"],))
    curve = [[e["step"], e["mean"]] for e in evals if e["purpose"] == "periodic"]
    final = next((e["mean"] for e in evals if e["purpose"] == "final"), None)
    means = [m for _, m in curve]
    cap = run["config"]["eval_max_score"]
    peak, drops = 0.0, 0
    for m in means:  # catastrophic degradation: falling below half of an already-good running best
        if peak >= 10 and m < 0.5 * peak:
            drops += 1
        peak = max(peak, m)
    wall = run["wall_seconds"] or 0.0
    return {"run_id": run["id"], "seed": run["seed"], "status": run["status"], "steps": run["steps"],
            "wall_seconds": wall, "sps": run["steps"] / wall if wall else None, "curve": curve,
            "final_mean": final, "best_mean": max(means) if means else None,
            "auc": float(np.mean(means) / cap) if means else None, "degradations": drops,
            "steps_to": {str(t): next((s for s, m in curve if m >= t), None) for t in THRESHOLDS},
            "model_id": run["model_id"]}


def _stat(values) -> dict | None:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "ci": bootstrap_ci(v), "std": float(np.std(v)), "values": v}


def group_summary(runs: list[dict]) -> dict:
    """Aggregate completed runs of one group. Curves are averaged on the common evaluation grid."""
    rs = [run_summary(r) for r in runs]
    steps = sorted({s for r in rs for s, _ in r["curve"]})
    curve = []
    for s in steps:
        vals = [m for r in rs for st, m in r["curve"] if st == s]
        lo, hi = bootstrap_ci(vals, 1000)
        curve.append({"step": s, "mean": float(np.mean(vals)), "lo": lo, "hi": hi, "n": len(vals)})
    finals = [r["final_mean"] for r in rs if r["final_mean"] is not None]
    first = runs[0]
    return {
        "group_key": first["group_key"], "label": group_label(first), "algo": first["algo"], "variant": first["variant"],
        "kind": first["kind"], "total_steps": first["total_steps"], "env_version": first["env_version"],
        "n_seeds": len(rs), "seeds": [r["seed"] for r in rs], "run_ids": [r["run_id"] for r in rs],
        "final": _stat(finals), "best": _stat([r["best_mean"] for r in rs]), "auc": _stat([r["auc"] for r in rs]),
        "seed_cv": float(np.std(finals) / max(np.mean(finals), 1e-9)) if finals else None,
        "degradations_mean": float(np.mean([r["degradations"] for r in rs])),
        "steps_to": {str(t): {"reached": sum(r["steps_to"][str(t)] is not None for r in rs), "of": len(rs),
                              "median": _median([r["steps_to"][str(t)] for r in rs])} for t in THRESHOLDS},
        "wall_seconds_mean": float(np.mean([r["wall_seconds"] for r in rs])),
        "sps_mean": float(np.mean([r["sps"] for r in rs if r["sps"]])) if any(r["sps"] for r in rs) else None,
        "curve": curve, "runs": rs,
    }


def _median(values):
    """Median steps-to-threshold, counting seeds that never reached it as +inf (None if most never did)."""
    v = sorted(float("inf") if x is None else x for x in values)
    m = v[(len(v) - 1) // 2]
    return None if m == float("inf") else m


def group_label(run: dict) -> str:
    if run["algo"] == "xdqn":
        return "xdqn" if run["variant"] in (None, "control") else f"xdqn+{run['variant']}"
    return run["algo"]


def groups(total_steps: int | None = None, env_version: str | None = None) -> list[dict]:
    sql, params = "SELECT * FROM runs WHERE status='completed' AND COALESCE(arm,'')<>'validation'", []
    if total_steps:
        sql, params = sql + " AND total_steps=?", params + [total_steps]
    if env_version:
        sql, params = sql + " AND env_version=?", params + [env_version]
    by_key: dict[str, list[dict]] = {}
    for run in db.query(sql + " ORDER BY id", params):
        by_key.setdefault(run["group_key"], []).append(run)
    return [group_summary(rs) for rs in by_key.values()]


def benchmarks() -> list[dict]:
    """Distinct benchmark settings. Rankings are only ever computed inside one of these."""
    return db.query("SELECT env_version, total_steps, COUNT(*) AS runs, COUNT(DISTINCT group_key) AS groups "
                    "FROM runs WHERE status='completed' AND COALESCE(arm,'')<>'validation' "
                    "GROUP BY env_version, total_steps ORDER BY runs DESC")


LEADERBOARD_METRICS = {
    "final_mean": "Mean validation score of the final policy (50 episodes), averaged over seeds",
    "best_mean": "Best periodic validation mean during training, averaged over seeds",
    "auc": "Normalised area under the validation learning curve (sample efficiency)",
    "steps_to_25": "Median environment steps to reach validation mean >= 25 (lower is better)",
    "steps_to_50": "Median environment steps to reach validation mean >= 50 (lower is better)",
    "steps_to_100": "Median environment steps to reach validation mean >= 100 (lower is better)",
    "test_mean": "Held-out test mean (confirmatory evaluations only)",
    "ood_mean": "Mean over out-of-distribution test scenarios (confirmatory evaluations only)",
}


def generalization(run_ids: list[int]) -> dict:
    """Confirmatory held-out results per scenario, averaged over the group's models."""
    if not run_ids:
        return {}
    marks = ",".join("?" * len(run_ids))
    rows = db.query(f"SELECT scenario, mean FROM evals WHERE purpose='confirmatory' AND split='test' "
                    f"AND run_id IN ({marks})", run_ids)
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["scenario"], []).append(r["mean"])
    return {k: _stat(v) for k, v in out.items()}


def leaderboard(metric: str, total_steps: int, env_version: str) -> list[dict]:
    if metric not in LEADERBOARD_METRICS:
        raise ValueError(f"metric must be one of {sorted(LEADERBOARD_METRICS)}")
    rows = []
    for g in groups(total_steps, env_version):
        gen = generalization(g["run_ids"])
        ood = [gen[s]["mean"] for s in OOD_SCENARIOS if s in gen]
        value = {
            "final_mean": g["final"] and g["final"]["mean"], "best_mean": g["best"] and g["best"]["mean"],
            "auc": g["auc"] and g["auc"]["mean"],
            "test_mean": gen["default"]["mean"] if "default" in gen else None,
            "ood_mean": float(np.mean(ood)) if ood else None,
        }.get(metric, g["steps_to"].get(metric.removeprefix("steps_to_"), {}).get("median"))
        rows.append({"label": g["label"], "group_key": g["group_key"], "algo": g["algo"], "variant": g["variant"],
                     "kind": g["kind"], "n_seeds": g["n_seeds"], "value": value,
                     "ci": {"final_mean": g["final"], "best_mean": g["best"], "auc": g["auc"]}.get(metric, {}) and
                           {"final_mean": g["final"], "best_mean": g["best"], "auc": g["auc"]}[metric]["ci"],
                     "final_mean": g["final"] and g["final"]["mean"], "wall_seconds_mean": g["wall_seconds_mean"],
                     "run_ids": g["run_ids"]})
    lower_better = metric.startswith("steps_to")
    ranked = sorted((r for r in rows if r["value"] is not None), key=lambda r: r["value"], reverse=not lower_better)
    unranked = [r for r in rows if r["value"] is None]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked + unranked
