"""Bounded autonomous research loop (a real process, not a simulated one).

    python -m backend.research.orchestration.controller SESSION_ID

One session repeats  OBSERVE -> HYPOTHESIZE -> IMPLEMENT(check) -> VALIDATE -> TRAIN -> EVALUATE ->
DECIDE -> DOCUMENT -> ITERATE  until a budget is exhausted, the queue is empty or the user stops it.
The controller executes and judges experiments; *writing new hypotheses/variants* is done by a coding
agent in separate, explicit sessions (see `write_brief` / `invoke_agent` and docs/research-methodology.md).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np
import torch

from backend.models import registry
from backend.research.hypotheses.library import seed_library
from backend.research.reporting import report
from backend.research.selection.decide import decide
from backend.rl.agents.dqn import DQNConfig
from backend.rl.environments.env import OBS_DIM, FlapEnv
from backend.rl.evaluation.evaluate import OOD_SCENARIOS, evaluate_model
from backend.rl.experimental import make_agent, parts, source_sha
from backend.rl.training import train
from backend.storage import db

STAGE_SEED0 = {"screening": 0, "intermediate": 50, "confirmation": 100}  # disjoint training seeds per stage
OVERRIDABLE = {"max_experiments": (0, 50), "steps": (1000, 2_000_000), "seeds": (1, 16),
               "session_wall_seconds": (30, 86_400), "max_concurrent": (1, 16), "max_memory_mb": (512, 32_768),
               "max_gpu_mem_mb": (256, 12_288), "max_disk_mb": (256, 102_400), "run_wall_seconds": (30, 86_400),
               "eval_every": (500, 1_000_000), "eval_episodes": (2, 200), "early_stop_patience": (0, 100),
               "checkpoint_every": (500, 5_000_000)}


def load_profiles() -> dict:
    with open(db.ROOT / "configs" / "profiles.toml", "rb") as f:
        return tomllib.load(f)


def budgets_for(profile: str, overrides: dict | None = None) -> dict:
    cfg = load_profiles()
    if profile not in cfg["profiles"]:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(cfg['profiles'])}")
    b = {**cfg["limits"], **cfg["profiles"][profile]}
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in OVERRIDABLE:
            raise ValueError(f"budget {k!r} cannot be overridden; allowed: {sorted(OVERRIDABLE)}")
        lo, hi = OVERRIDABLE[k]
        if not lo <= v <= hi:
            raise ValueError(f"budget {k} must be within {lo}..{hi}")
        b[k] = v
    return b


def disk_usage_mb() -> float:
    total = 0
    for root in (db.experiments_dir(), db.checkpoints_dir()):
        for dirpath, _, files in os.walk(root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    return total / 2**20


def run_limits(b: dict) -> dict:
    return {"max_wall_seconds": b["run_wall_seconds"], "max_memory_mb": b["max_memory_mb"],
            "max_gpu_mem_mb": b["max_gpu_mem_mb"], "early_stop_patience": b["early_stop_patience"]}


# ---------------------------------------------------------------- sessions
def create_session(profile: str, objective: str = "", overrides: dict | None = None) -> int:
    db.init()
    seed_library()
    return db.insert("sessions", profile=profile, objective=objective[:2000], budgets=budgets_for(profile, overrides),
                     status="queued", control="run", started_at=time.time(), experiments_done=0, notes="")


def launch_session(session_id: int) -> subprocess.Popen:
    logs = db.experiments_dir() / "logs"
    logs.mkdir(exist_ok=True)
    with open(logs / f"session_{session_id:04d}.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "backend.research.orchestration.controller", str(session_id)],
                                cwd=db.ROOT, stdout=log, stderr=subprocess.STDOUT)
    db.update("sessions", session_id, pid=proc.pid)
    return proc


def set_control(session_id: int, control: str) -> None:
    if control not in ("run", "pause", "stop"):
        raise ValueError("control must be run, pause or stop")
    db.update("sessions", session_id, control=control)


def active_session() -> dict | None:
    return db.one("SELECT * FROM sessions WHERE status IN ('queued','running','paused') ORDER BY id DESC")


def reconcile() -> None:
    import psutil
    for s in db.query("SELECT * FROM sessions WHERE status IN ('queued','running','paused')"):
        alive = False
        try:
            alive = s["pid"] and "orchestration.controller" in " ".join(psutil.Process(s["pid"]).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        if not alive:
            db.update("sessions", s["id"], status="interrupted", finished_at=time.time(),
                      error="controller process disappeared")
            db.query("UPDATE experiments SET status='interrupted' WHERE session_id=? AND status='running'", (s["id"],))


# ---------------------------------------------------------------- OBSERVE
def observe(b: dict) -> dict:
    """Summarise what the stored evidence says about the current control agent."""
    all_groups = report.groups()
    control = [g for g in all_groups if g["algo"] == "xdqn" and g["variant"] == "control"]
    same = [g for g in control if g["total_steps"] == b["steps"]]
    g = max(same or control, key=lambda x: x["n_seeds"], default=None)
    weaknesses, evidence = [], {}
    if g and g["final"]:
        cap = 200
        ppo = max((x for x in all_groups if x["algo"] == "ppo" and x["total_steps"] == g["total_steps"]),
                  key=lambda x: x["n_seeds"], default=None)
        evidence = {"control_group": g["group_key"], "total_steps": g["total_steps"], "n_seeds": g["n_seeds"],
                    "final_mean": g["final"]["mean"], "auc": g["auc"]["mean"], "seed_cv": g["seed_cv"],
                    "degradations_mean": g["degradations_mean"], "ppo_auc": ppo and ppo["auc"]["mean"],
                    "ppo_final_mean": ppo and ppo["final"]["mean"]}
        if g["degradations_mean"] >= 0.5:
            weaknesses.append({"name": "instability", "severity": float(g["degradations_mean"]),
                               "evidence": f"{g['degradations_mean']:.1f} catastrophic validation drops per run"})
        if g["seed_cv"] is not None and g["seed_cv"] >= 0.25:
            weaknesses.append({"name": "seed_variance", "severity": float(g["seed_cv"]) * 2,
                               "evidence": f"final-policy mean varies across seeds (CV {g['seed_cv']:.2f})"})
        ref_auc = max(0.6, (ppo["auc"]["mean"] if ppo else 0.0))
        if g["auc"]["mean"] < ref_auc - 0.05:
            weaknesses.append({"name": "sample_efficiency", "severity": float(ref_auc - g["auc"]["mean"]) * 4,
                               "evidence": f"validation AUC {g['auc']['mean']:.2f} vs reference {ref_auc:.2f}"})
        if g["final"]["mean"] < 0.9 * cap:
            weaknesses.append({"name": "final_performance", "severity": float(1 - g["final"]["mean"] / cap) * 2,
                               "evidence": f"final-policy mean {g['final']['mean']:.0f} of cap {cap}"})
        gen = report.generalization(g["run_ids"])
        ood = [gen[s]["mean"] for s in OOD_SCENARIOS if s in gen]
        if ood and "default" in gen and np.mean(ood) < 0.7 * gen["default"]["mean"]:
            weaknesses.append({"name": "generalization", "severity": 1.0,
                               "evidence": f"OOD mean {np.mean(ood):.0f} vs in-distribution {gen['default']['mean']:.0f}"})
    weaknesses.sort(key=lambda w: -w["severity"])
    return {"weaknesses": weaknesses, "evidence": evidence,
            "note": None if g else "no completed control runs yet: the first experiment will create them"}


# ---------------------------------------------------------------- HYPOTHESIZE / ITERATE
def queue(obs: dict, b: dict) -> list[dict]:
    """Ranked next experiments. Follow-up stages come first; rejected/inconclusive hypotheses are never repeated."""
    severity = {w["name"]: w["severity"] for w in obs["weaknesses"]}
    items = []
    for h in db.query("SELECT * FROM hypotheses ORDER BY id"):
        if h["status"] == "promising" and b.get("confirm"):
            items.append((0, h, "confirmation", "promising at an exploratory stage: needs fresh-seed confirmation"))
        elif h["status"] == "requires_further_testing" and not b.get("confirm_only"):
            items.append((1, h, "intermediate", "favourable but non-significant screening trend: add seeds"))
        elif h["status"] == "proposed" and not b.get("confirm_only"):
            hit = [t for t in (h["targets"] or "").split(",") if t in severity]
            score = sum(severity[t] for t in hit)
            why = (f"targets observed weakness(es): {', '.join(hit)}" if hit else
                   "untested; no matching weakness observed yet")
            items.append((2 - min(score, 100) / 1000, h, "screening", why))
    items.sort(key=lambda x: (x[0], x[1]["id"]))
    return [{"hypothesis_id": h["id"], "title": h["title"], "variant": h["variant"], "stage": stage, "reason": why}
            for _, h, stage, why in items]


# ---------------------------------------------------------------- VALIDATE
def preflight(variant: str) -> None:
    """In-process checks: construction, rollout, tensor shapes, finiteness, checkpoint round-trip. Raises on failure."""
    if True:
        parts(variant)
        cfg = DQNConfig(n_envs=2, buffer_size=2000, batch_size=32, learning_starts=64)
        agent = make_agent(variant, cfg, 2000, "cpu", 0)
        envs = [FlapEnv() for _ in range(2)]
        obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)])
        for step in range(0, 600, 2):
            actions = agent.act(obs, step)
            assert actions.shape == (2,) and set(np.unique(actions)) <= {0, 1}, "bad action shape/values"
            nxt, loop = np.empty_like(obs), np.empty_like(obs)
            rew, done, ended = np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(2, bool)
            for i, e in enumerate(envs):
                o2, r, term, trunc, info = e.step(int(actions[i]))
                nxt[i], rew[i], done[i], ended[i] = o2, r, term, term or trunc
                if ended[i]:
                    agent.on_episode(info["score"], e.world.tick)
                    o2 = e.reset()[0]
                loop[i] = o2
            agent.store(obs, actions, rew, nxt, done, ended)
            obs = loop
            if step >= cfg.learning_starts:
                m = agent.update(step)
                assert all(np.isfinite(v) for v in m.values()), f"non-finite update metrics {m}"
        probe = np.random.default_rng(0).random((16, OBS_DIM), np.float32)
        q = agent.q_values(probe)
        assert q.shape == (16, 2) and np.isfinite(q).all(), "Q output must be finite [B, 2]"
        clone = make_agent(variant, cfg, 2000, "cpu", 1)
        clone.load_state(agent.state())
        assert np.allclose(clone.q_values(probe), q), "checkpoint round-trip changed the policy"
        with torch.no_grad():
            assert all(torch.isfinite(p).all() for p in agent.net.parameters()), "non-finite weights"


def validate_variant(variant: str, b: dict) -> tuple[bool, str]:
    """Cheap pre-flight before any expensive training. Returns (ok, detail)."""
    try:
        preflight(variant)
    except Exception:
        return False, traceback.format_exc()[-1500:]
    # tiny real training run through the production trainer (subprocess, checkpoint + registry path included)
    rid = train.create_run("xdqn", variant, seed=9000, total_steps=3000, profile="validation", arm="validation",
                           eval_every=1500, eval_episodes=2, checkpoint_every=1500,
                           hyper={"learning_starts": 500}, limits={**run_limits(b), "max_wall_seconds": 120})
    status = train.run_batch([rid], 1)[rid]
    if status != "completed":
        err = db.one("SELECT error FROM runs WHERE id=?", (rid,))["error"]
        return False, f"training smoke run {rid} ended as {status}: {(err or '')[-1200:]}"
    registry.load(registry.model_id(db.one("SELECT * FROM runs WHERE id=?", (rid,))))  # loadable via the registry
    db.update("models", registry.model_id({"algo": "xdqn", "variant": variant, "id": rid}), status="archived")
    return True, f"rollout, shapes, finiteness, checkpoint round-trip and smoke run {rid} passed"


# ---------------------------------------------------------------- TRAIN / EVALUATE / DECIDE / DOCUMENT
def _arm_runs(variant: str, seeds: list[int], steps: int, b: dict, exp_id: int, arm: str, profile: str):
    new, ids = [], []
    for seed in seeds:
        rid = train.create_run("xdqn", variant, seed=seed, total_steps=steps, profile=profile, experiment_id=exp_id,
                               arm=arm, eval_every=b["eval_every"], eval_episodes=b["eval_episodes"],
                               checkpoint_every=b["checkpoint_every"], limits=run_limits(b))
        key = db.one("SELECT config_key FROM runs WHERE id=?", (rid,))["config_key"]
        done = db.one("SELECT id FROM runs WHERE config_key=? AND status='completed' AND id<>? ORDER BY id", (key, rid))
        if done:  # scientifically identical run already exists: reuse it instead of recomputing
            with db.tx() as con:
                con.execute("DELETE FROM runs WHERE id=?", (rid,))
            ids.append(done["id"])
        else:
            ids.append(rid)
            new.append(rid)
    return ids, new


def _arm_metrics(run_ids: list[int]) -> dict:
    marks = ",".join("?" * len(run_ids))
    runs = db.query(f"SELECT * FROM runs WHERE id IN ({marks})", run_ids)
    done = [report.run_summary(r) for r in runs if r["status"] == "completed"]
    return {"completed": len(done), "launched": len(run_ids),
            "not_completed": {r["id"]: r["status"] for r in runs if r["status"] != "completed"},
            "final": [r["final_mean"] for r in done], "auc": [r["auc"] for r in done],
            "degradations": [r["degradations"] for r in done], "wall": [r["wall_seconds"] for r in done],
            "steps_to_50": [r["steps_to"]["50"] for r in done], "run_ids": [r["run_id"] for r in done]}


def _held_out(run_ids: list[int]) -> dict:
    """Confirmation only: evaluate every model of an arm on the held-out test split, in- and out-of-distribution."""
    out: dict[str, list[float]] = {}
    for rid in run_ids:
        mid = db.one("SELECT model_id FROM runs WHERE id=?", (rid,))["model_id"]
        if db.one("SELECT 1 AS x FROM evals WHERE model_id=? AND purpose='confirmatory'", (mid,)):
            continue
        evaluate_model(mid, "test", ["default", *OOD_SCENARIOS], 50, "confirmatory")
    gen = report.generalization(run_ids)
    for scenario, stat in gen.items():
        out[scenario] = stat["values"]
    return out


def consider_promotion(run_ids: list[int]) -> str:
    """Champion (chosen on validation) of a confirmed arm replaces the best verified model only if it also
    beats it on the held-out test evidence."""
    marks = ",".join("?" * len(run_ids))
    champ = db.one(f"SELECT run_id, model_id, mean FROM evals WHERE purpose='final' AND run_id IN ({marks}) "
                   f"ORDER BY mean DESC, run_id", run_ids)
    if not champ:
        return "no candidate"

    def test_score(mid):
        rows = db.query("SELECT id, scenario, mean FROM evals WHERE model_id=? AND purpose='confirmatory' "
                        "AND split='test'", (mid,))
        if not rows:
            return None
        default = next((r for r in rows if r["scenario"] == "default"), None)
        ood = [r["mean"] for r in rows if r["scenario"] != "default"]
        return (default["mean"] if default else 0.0, float(np.mean(ood)) if ood else 0.0, default and default["id"])

    cand = test_score(champ["model_id"])
    if cand is None:
        return "candidate has no held-out evaluation"
    best = registry.best_verified()
    inc = test_score(best["id"]) if best else None
    if best and inc and cand[:2] <= inc[:2]:
        return f"kept {best['id']} (test {inc[0]:.1f}/OOD {inc[1]:.1f}) over {champ['model_id']} (test {cand[0]:.1f}/OOD {cand[1]:.1f})"
    registry.promote(champ["model_id"], {"test_eval_id": cand[2], "test_mean": cand[0], "ood_mean": cand[1],
                                         "replaced": best["id"] if best else None})
    return f"promoted {champ['model_id']} (test {cand[0]:.1f}/OOD {cand[1]:.1f})"


def write_report(exp_id: int) -> str:
    e = db.one("SELECT * FROM experiments WHERE id=?", (exp_id,))
    h = db.one("SELECT * FROM hypotheses WHERE id=?", (e["hypothesis_id"],))
    r, c = e["result"] or {}, e["config"] or {}
    lines = [f"# Experiment {exp_id}: {h['id']} — {h['title']}", "",
             f"- Stage: **{e['stage']}** ({'confirmatory' if e['stage'] == 'confirmation' else 'exploratory'})"
             f" · profile `{e['profile']}` · code `{e['code_version']}` · variant source `{c.get('variant_source_sha')}`",
             f"- Decision: **{e['decision']}**", f"- Conclusion: {e['conclusion']}", "",
             "## Hypothesis", f"- Area: {h['area']}", f"- Motivation: {h['motivation']}",
             f"- Modification: {h['modification']}", f"- Expected benefit: {h['expected_benefit']}",
             f"- Failure modes considered: {h['failure_modes']}", f"- Prior work: {h['prior_work']}",
             f"- Novelty status: {h['novelty']}", f"- Baseline: {h['baseline']}", f"- Criteria: {h['criteria']}", "",
             "## Setup", f"- Variant `{c.get('variant')}` vs `control`, {c.get('steps')} env steps per run, "
             f"training seeds {c.get('seeds')}", f"- Validation: {c.get('eval_episodes')} fixed episodes every "
             f"{c.get('eval_every')} steps; final policy on 50 episodes", f"- Runs: {c.get('arms')}",
             f"- Reused identical control runs: {c.get('reused')}", f"- Pre-flight validation: {r.get('validation')}", ""]
    if r.get("arms"):
        lines += ["## Results (per training seed)", "", "| arm | completed | final mean | AUC | degradations | wall s |",
                  "|---|---|---|---|---|---|"]
        for arm, m in r["arms"].items():
            f = lambda v: "–" if not v else f"{np.mean(v):.3g} ({', '.join(f'{x:.3g}' for x in v)})"
            lines.append(f"| {arm} | {m['completed']}/{m['launched']} | {f(m['final'])} | {f(m['auc'])} | "
                         f"{f(m['degradations'])} | {f(m['wall'])} |")
        lines.append("")
    for k in ("auc", "final_mean"):
        if r.get(k):
            s = r[k]
            lines.append(f"- {k}: candidate {s['mean_a']:.3g} vs control {s['mean_b']:.3g}; diff {s['diff']:+.3g}, "
                         f"95% CI [{s['ci'][0]:+.3g}, {s['ci'][1]:+.3g}], P(improvement) {s['prob_improvement']:.2f}")
    if r.get("wall_ratio"):
        lines.append(f"- Wall-clock ratio candidate/control: {r['wall_ratio']:.2f} (>1 = candidate is more expensive)")
    if r.get("held_out"):
        lines += ["", "## Held-out test (confirmation stage only)", "", "| scenario | candidate | control |", "|---|---|---|"]
        for sc in ["default", *OOD_SCENARIOS]:
            a, b_ = r["held_out"]["candidate"].get(sc), r["held_out"]["control"].get(sc)
            lines.append(f"| {sc} | {np.mean(a):.1f} | {np.mean(b_):.1f} |" if a and b_ else f"| {sc} | – | – |")
        lines.append(f"\nPromotion: {r.get('promotion')}")
    lines += ["", "## Limitations", "- Bootstrap CIs over a handful of seeds are coarse; exploratory stages are not evidence.",
              "- CPU training is not bit-reproducible across machines/BLAS builds; seeds control everything else.",
              "- The score cap (200) saturates strong agents; differences above it are invisible."]
    path = db.experiments_dir() / "reports" / f"exp_{exp_id:04d}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    db.update("experiments", exp_id, report_path=db.rel(path))
    return str(path)


def run_experiment(session: dict, item: dict, b: dict, should_stop) -> dict:
    h = db.one("SELECT * FROM hypotheses WHERE id=?", (item["hypothesis_id"],))
    stage, variant = item["stage"], h["variant"]
    conf = load_profiles()["profiles"]["final_confirmation"]
    steps = conf["steps"] if stage == "confirmation" else b["steps"]
    n = max(b["seeds"], conf["seeds"]) if stage == "confirmation" else b["seeds"]
    seeds = list(range(STAGE_SEED0[stage], STAGE_SEED0[stage] + n))
    cfg = {"variant": variant, "stage": stage, "steps": steps, "seeds": seeds, "eval_every": b["eval_every"],
           "eval_episodes": b["eval_episodes"], "observations": item.get("observations"), "reason": item["reason"]}
    exp_id = db.insert("experiments", session_id=session["id"], hypothesis_id=h["id"], stage=stage,
                       profile=session["profile"], status="running", config=cfg, code_version=db.code_version(),
                       started_at=time.time())
    db.update("hypotheses", h["id"], budget={"steps": steps, "seeds": n}, updated_at=time.time())

    def close(status, decision, conclusion, result, hyp_status=None, error=None):
        db.update("experiments", exp_id, status=status, decision=decision, conclusion=conclusion, result=result,
                  finished_at=time.time(), error=error, config=cfg)
        if hyp_status and b.get("record_status", True):  # wiring-check profiles never change a hypothesis
            db.update("hypotheses", h["id"], status=hyp_status, updated_at=time.time())
        write_report(exp_id)
        return db.one("SELECT * FROM experiments WHERE id=?", (exp_id,))

    try:
        try:
            cfg["variant_source_sha"] = source_sha(variant)  # IMPLEMENT: the isolated variant module must exist
            ok, detail = validate_variant(variant, b)        # VALIDATE
        except Exception:
            ok, detail = False, traceback.format_exc()[-1500:]
        if not ok:
            db.event("warning", "research", f"{h['id']} failed pre-flight validation", None)
            return close("failed", "inconclusive", "Implementation failed pre-flight validation; no training was run.",
                         {"validation": detail}, "implementation_failed", detail)
        control_ids, new_c = _arm_runs("control", seeds, steps, b, exp_id, "control", session["profile"])
        cand_ids, new_v = _arm_runs(variant, seeds, steps, b, exp_id, "candidate", session["profile"])
        if stage == "intermediate":  # pool with the screening seeds of the same budget
            prev = db.one("SELECT config FROM experiments WHERE hypothesis_id=? AND stage='screening' "
                          "AND status='completed' ORDER BY id DESC", (h["id"],))
            if prev and prev["config"]["steps"] == steps:
                control_ids += prev["config"]["arms"]["control"]
                cand_ids += prev["config"]["arms"]["candidate"]
        cfg["arms"] = {"control": control_ids, "candidate": cand_ids}
        cfg["reused"] = sorted(set(control_ids + cand_ids) - set(new_c + new_v))
        db.update("experiments", exp_id, config=cfg)
        train.run_batch(new_c + new_v, b["max_concurrent"], should_stop)   # TRAIN
        arms = {"control": _arm_metrics(control_ids), "candidate": _arm_metrics(cand_ids)}  # EVALUATE
        result = {"validation": detail, "arms": arms}
        if should_stop():
            return close("stopped", None, "Stopped by the user before completion; no decision recorded.", result)
        if min(arms["control"]["completed"], arms["candidate"]["completed"]) < 1:
            return close("failed", "inconclusive", "An arm produced no completed runs; see run errors.", result,
                         "inconclusive", str(arms["candidate"]["not_completed"]))
        result["auc"] = report.compare(arms["candidate"]["auc"], arms["control"]["auc"])
        result["final_mean"] = report.compare(arms["candidate"]["final"], arms["control"]["final"])
        result["wall_ratio"] = float(np.mean(arms["candidate"]["wall"]) / max(np.mean(arms["control"]["wall"]), 1e-9))
        decision, why = decide(stage, result["auc"], result["final_mean"])                   # DECIDE
        if result["wall_ratio"] > 1.5 and decision in ("promising", "confirmed"):
            why += (f" Note: {result['wall_ratio']:.1f}x the control's wall-clock per run — more sample-efficient, "
                    "not more compute-efficient.")
        if stage == "confirmation":
            result["held_out"] = {"candidate": _held_out(arms["candidate"]["run_ids"]),
                                  "control": _held_out(arms["control"]["run_ids"])}
            result["promotion"] = (consider_promotion(arms["candidate"]["run_ids"]) if decision == "confirmed"
                                   else "not considered (not confirmed)")
        return close("completed", decision, why, result, decision)                            # DOCUMENT
    except Exception:
        err = traceback.format_exc()
        db.event("error", "research", err[-3000:])
        return close("failed", "inconclusive", "The experiment crashed; recorded as a failure, not as evidence.",
                     {"error": err[-1500:]}, "errored", err[-3000:])  # not re-queued automatically


# ---------------------------------------------------------------- coding-agent hand-off
def write_brief(obs: dict) -> str:
    """Research brief for the next coding-agent session: what is known, what failed, what to propose next."""
    hyps = db.query("SELECT id, title, variant, status FROM hypotheses ORDER BY id")
    exps = db.query("SELECT id, hypothesis_id, stage, decision, conclusion FROM experiments ORDER BY id")
    text = ["# Research brief", "", "You are the research agent of this lab. Read docs/research-methodology.md first.",
            "", "## Current observations", *[f"- {w['name']}: {w['evidence']}" for w in obs["weaknesses"]],
            f"- evidence: {obs['evidence']}", "", "## Hypotheses so far",
            *[f"- {h['id']} `{h['variant']}` — {h['status']}: {h['title']}" for h in hyps], "", "## Experiments",
            *[f"- exp {e['id']} {e['hypothesis_id']} [{e['stage']}] {e['decision']}: {e['conclusion']}" for e in exps],
            "", "## Your task",
            "Propose 1-3 NEW hypotheses that are meaningfully different from every entry above (do not re-run "
            "rejected ideas with new hyperparameters). For each: add one module under backend/rl/experimental/ "
            "that overrides DQNAgent hooks, register it in VARIANTS, append a complete entry to "
            "backend/research/hypotheses/library.py (cite the closest prior work; novelty 'known' or 'unverified'), "
            "and run `python -m pytest tests -q`. Do not start training; the controller does that under its budgets."]
    path = db.experiments_dir() / "research_brief.md"
    path.write_text("\n".join(text) + "\n")
    return str(path)


def invoke_agent(brief_path: str) -> str:
    """Optional, opt-in: run the configured coding-agent command with the brief on stdin, once, with a timeout.
    Set FLAPLAB_AGENT_CMD (e.g. `claude -p --permission-mode acceptEdits`). Unset = no agent is invoked."""
    cmd = os.environ.get("FLAPLAB_AGENT_CMD")
    if not cmd:
        return "no agent command configured (FLAPLAB_AGENT_CMD unset); brief written for the next manual session"
    try:
        with open(brief_path) as f:
            p = subprocess.run(shlex.split(cmd), stdin=f, capture_output=True, text=True, cwd=db.ROOT,
                               timeout=int(os.environ.get("FLAPLAB_AGENT_TIMEOUT", "1800")))
        msg = f"agent command exited {p.returncode}: {(p.stdout or p.stderr)[-500:]}"
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"agent command failed: {exc}"
    db.event("info", "research", msg)
    return msg


# ---------------------------------------------------------------- session loop
def run_session(session_id: int) -> str:
    db.init()
    seed_library()
    s = db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
    if s is None:
        raise SystemExit(f"session {session_id} does not exist")
    b, t0, notes = s["budgets"], time.time(), []
    db.update("sessions", session_id, status="running", pid=os.getpid(), started_at=t0)

    def control() -> str:
        return db.one("SELECT control FROM sessions WHERE id=?", (session_id,))["control"]

    def finish(status: str, why: str) -> str:
        notes.append(why)
        if os.environ.get("FLAPLAB_AUTO_CURATE", "1") != "0":
            registry.curate()  # new checkpoints: the list keeps only each group's best, under its curated name
        db.update("sessions", session_id, status=status, finished_at=time.time(), notes="\n".join(notes))
        return status

    try:
        while True:
            ctl = control()
            if ctl == "stop":
                return finish("stopped", "Stopped by the user; no further experiments were launched.")
            if ctl == "pause":
                db.update("sessions", session_id, status="paused")
                time.sleep(1.0)
                continue
            db.update("sessions", session_id, status="running")
            done = db.one("SELECT experiments_done FROM sessions WHERE id=?", (session_id,))["experiments_done"]
            if done >= b["max_experiments"]:
                return finish("completed", f"Experiment budget reached ({done}/{b['max_experiments']}).")
            if time.time() - t0 > b["session_wall_seconds"]:
                return finish("completed", f"Session wall-clock budget reached ({b['session_wall_seconds']} s).")
            used = disk_usage_mb()
            if used > b["max_disk_mb"]:
                db.event("warning", "research", f"disk budget exceeded: {used:.0f} MB > {b['max_disk_mb']} MB")
                return finish("stopped", f"Disk budget exceeded ({used:.0f} MB of {b['max_disk_mb']} MB).")
            obs = observe(b)                                       # OBSERVE
            q = queue(obs, b)                                      # HYPOTHESIZE / select
            if not q:
                brief = write_brief(obs)
                agent = invoke_agent(brief) if b.get("invoke_agent") else "agent invocation not enabled for this session"
                return finish("completed", f"Hypothesis queue exhausted. Brief: {brief}. {agent}")
            item = {**q[0], "observations": obs}
            exp = run_experiment({**s, "id": session_id}, item, b, lambda: control() == "stop")
            notes.append(f"exp {exp['id']} {exp['hypothesis_id']} [{exp['stage']}] -> {exp['decision'] or exp['status']}")
            db.update("sessions", session_id, experiments_done=done + 1, notes="\n".join(notes))
    except Exception:
        err = traceback.format_exc()
        db.update("sessions", session_id, status="failed", finished_at=time.time(), error=err[-3000:])
        db.event("error", "research", err[-3000:])
        raise


if __name__ == "__main__":
    run_session(int(sys.argv[1]))
