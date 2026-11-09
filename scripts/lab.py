"""Command-line entry points.  python -m scripts.lab <command>

  serve [--port 8000]            start the API (and the built frontend, if frontend/dist exists)
  bench                          measure CPU vs CUDA training throughput and cache the choice
  baselines [--profile P]        train the DQN / PPO / xdqn reference groups of a profile
  confirm-baselines              held-out test + out-of-distribution evaluation of baseline champions
  research [--profile P] [--max-experiments N]   run one bounded research session in the foreground
  curate                         list only the best checkpoint per algorithm/variant, under names like PPO_Baseline
  status                         print a short summary of the lab database
"""
from __future__ import annotations

import argparse
import json

from backend.models import registry
from backend.research.orchestration import controller
from backend.research.reporting import report
from backend.rl.evaluation.evaluate import OOD_SCENARIOS, evaluate_model
from backend.rl.training import train
from backend.storage import db


def baselines(profile: str) -> None:
    b = controller.budgets_for(profile)
    ids = []
    for algo in ("dqn", "ppo", "xdqn"):
        for seed in range(b["seeds"]):
            rid = train.create_run(algo, seed=seed, total_steps=b["steps"], profile=profile, arm="baseline",
                                   eval_every=b["eval_every"], eval_episodes=b["eval_episodes"],
                                   checkpoint_every=b["checkpoint_every"], limits=controller.run_limits(b))
            key = db.one("SELECT config_key FROM runs WHERE id=?", (rid,))["config_key"]
            if db.one("SELECT id FROM runs WHERE config_key=? AND status='completed'", (key,)):
                with db.tx() as con:
                    con.execute("DELETE FROM runs WHERE id=?", (rid,))  # already benchmarked
            else:
                ids.append(rid)
    print(f"launching {len(ids)} runs (max {b['max_concurrent']} at once)")
    print(train.run_batch(ids, b["max_concurrent"]))
    curate()
    summary()


def confirm_baselines() -> None:
    """Pick each baseline group's champion on VALIDATION, evaluate it once on the held-out test split."""
    for g in report.groups():
        if g["kind"] != "baseline":
            continue
        champ = max(g["runs"], key=lambda r: (r["final_mean"] or -1, r["auc"] or -1))
        mid = champ["model_id"]
        if not db.one("SELECT 1 AS x FROM evals WHERE model_id=? AND purpose='confirmatory'", (mid,)):
            res = evaluate_model(mid, "test", ["default", *OOD_SCENARIOS], 50, "confirmatory")
            print(mid, {r["scenario"]: round(r["mean"], 1) for r in res})
        print(" ", controller.consider_promotion([champ["run_id"]]))


def curate() -> None:
    for c in registry.curate():
        print(f"{c['name']:<22} run {c['run_id']:<4} seed {c['seed']:<4} final={c['final_mean']:.1f} "
              f"auc={c['auc']:.3f} collapses={c['degradations']}  (best of {c['group_seeds']} seeds, {c['label']})")
    print("listed models:", len(registry.list_models()), "of", len(registry.list_models(include_unlisted=True)))


def summary() -> None:
    for g in report.groups():
        f, a = g["final"], g["auc"]
        print(f"{g['label']:<22} steps={g['total_steps']:<8} seeds={g['n_seeds']} "
              f"final={f['mean']:.1f} [{f['ci'][0]:.1f},{f['ci'][1]:.1f}] auc={a['mean']:.3f} "
              f"drops={g['degradations_mean']:.1f} wall={g['wall_seconds_mean']:.0f}s")
    best = db.kv_get("best_verified")
    print("best verified:", best and best["model_id"])
    for h in db.query("SELECT id, variant, status FROM hypotheses ORDER BY id"):
        print(f"  {h['id']} {h['variant']:<16} {h['status']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["serve", "bench", "baselines", "confirm-baselines", "research", "curate", "status"])
    ap.add_argument("--profile")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-experiments", type=int)
    args = ap.parse_args()
    db.init()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("backend.api.app:app", host=args.host, port=args.port, log_level="info")
    elif args.command == "bench":
        print(json.dumps(train.benchmark_devices(), indent=2))
    elif args.command == "baselines":
        baselines(args.profile or "baseline_benchmark")
    elif args.command == "confirm-baselines":
        confirm_baselines()
    elif args.command == "curate":
        curate()
    elif args.command == "research":
        sid = controller.create_session(args.profile or "experimental_screening", "cli session",
                                        {"max_experiments": args.max_experiments})
        print("session", sid, controller.run_session(sid))
        summary()
    else:
        summary()


if __name__ == "__main__":
    main()
