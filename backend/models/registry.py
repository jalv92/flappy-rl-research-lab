"""Model registry + the single inference path used by the game server and evaluation.

Statuses: training (run in progress) | experimental (unfinished budget or unconfirmed variant) |
evaluated (finished budget + final validation eval) | promoted (the one best verified model) |
archived | failed. Promotion only happens through `promote()`, which requires held-out evidence.
"""
from __future__ import annotations

import functools
import os
import time
from pathlib import Path

import numpy as np
import torch

from backend.rl.environments.env import ENV_VERSION, N_ACTIONS, OBS_DIM, OBS_VERSION
from backend.storage import db

STATUSES = ("training", "experimental", "evaluated", "promoted", "archived", "failed")


class IncompatibleModel(Exception):
    pass


def derived_id(run: dict) -> str:
    variant = run.get("variant")
    tag = f"-{variant}" if variant and variant != "control" else ""
    return f"{run['algo']}{tag}-r{run['id']}"


def model_id(run: dict) -> str:
    """The run's model id: its curated name once it has one, otherwise `<algo>[-variant]-r<run>`."""
    return run.get("model_id") or derived_id(run)


def champion_name(algo: str, variant: str | None) -> str:
    """Curated name of a group's best model: PPO_Baseline, DQN_Baseline, XDQN_Control, XDQN_PriorExplore..."""
    if algo != "xdqn":
        return f"{algo.upper()}_Baseline"
    if variant in (None, "control"):
        return "XDQN_Control"
    return "XDQN_" + "".join(w.capitalize() for part in variant.split("+") for w in part.split("_"))


def register(run: dict, checkpoint_path: str | Path, status: str, train_steps: int, metrics: dict | None = None) -> str:
    mid = model_id(run)
    prev = db.one("SELECT created_at, status, listed FROM models WHERE id=?", (mid,))
    if prev and prev["status"] in ("promoted", "archived"):
        status = prev["status"]  # never silently demote/replace a verified or retired model
    db.insert("models", id=mid, run_id=run["id"], algo=run["algo"], variant=run.get("variant"),
              experiment_id=run.get("experiment_id"), seed=run["seed"],
              created_at=prev["created_at"] if prev else time.time(), updated_at=time.time(),
              code_version=db.code_version(), config=run["config"], train_steps=train_steps, metrics=metrics or {},
              checkpoint_path=db.rel(checkpoint_path), obs_dim=OBS_DIM, obs_version=OBS_VERSION, n_actions=N_ACTIONS,
              env_version=ENV_VERSION, status=status, listed=prev["listed"] if prev else 1)
    db.update("runs", run["id"], model_id=mid)
    return mid


def get(mid: str) -> dict | None:
    return db.one("SELECT * FROM models WHERE id=?", (mid,))


def list_models(status: str | None = None, algo: str | None = None, include_unlisted: bool = False) -> list[dict]:
    """Registry rows. By default only the curated list: each group's champion, the promoted model and models of
    runs still training. `include_unlisted` returns every checkpoint ever registered."""
    sql, params = "SELECT * FROM models WHERE 1=1" + ("" if include_unlisted else " AND listed=1"), []
    if status:
        sql, params = sql + " AND status=?", params + [status]
    if algo:
        sql, params = sql + " AND algo=?", params + [algo]
    return db.query(sql + " ORDER BY created_at DESC", params)


def set_status(mid: str, status: str) -> None:
    if status not in STATUSES or status == "promoted":
        raise ValueError("use promote() for promotion; status must be one of " + ", ".join(STATUSES))
    db.update("models", mid, status=status, updated_at=time.time())


def best_verified() -> dict | None:
    return db.one("SELECT * FROM models WHERE status='promoted'")


def promote(mid: str, evidence: dict) -> None:
    """Make `mid` the best verified model. Caller must pass the held-out test evidence that justifies it."""
    if not evidence.get("test_eval_id"):
        raise ValueError("promotion requires a held-out test evaluation")
    with db.tx() as con:
        con.execute("UPDATE models SET status='evaluated', updated_at=? WHERE status='promoted'", (time.time(),))
        con.execute("UPDATE models SET status='promoted', updated_at=? WHERE id=?", (time.time(), mid))
    db.kv_set("best_verified", {"model_id": mid, "evidence": evidence, "at": time.time()})
    db.event("info", "registry", f"promoted {mid}: {evidence}")


def rename(old: str, new: str) -> None:
    """Rename a model everywhere it is referenced (runs, evaluations, stored matches and their replay files)."""
    if old == new:
        return
    if get(old) is None:
        raise KeyError(f"unknown model {old}")
    if get(new) is not None:
        raise ValueError(f"a model named {new} already exists")
    a, b = f'"{old}"', f'"{new}"'  # exact JSON string token: "xdqn-r1" never matches "xdqn-r11"
    replays = []
    with db.tx() as con:
        con.execute("UPDATE models SET id=?, updated_at=? WHERE id=?", (new, time.time(), old))
        con.execute("UPDATE runs SET model_id=? WHERE model_id=?", (new, old))
        con.execute("UPDATE evals SET model_id=? WHERE model_id=?", (new, old))
        for m in con.execute("SELECT id, participants, result, replay_path FROM matches").fetchall():
            if a in m["participants"]:
                con.execute("UPDATE matches SET participants=?, result=? WHERE id=?",
                            (m["participants"].replace(a, b), m["result"].replace(a, b), m["id"]))
                replays.append(m["replay_path"])
    for path in replays:
        if path and db.resolve(path).exists():
            db.resolve(path).write_text(db.resolve(path).read_text().replace(a, b))
    best = db.kv_get("best_verified")
    if best and best.get("model_id") == old:
        db.kv_set("best_verified", {**best, "model_id": new})
    _load.cache_clear()


def curate() -> list[dict]:
    """Keep the model list short: per algorithm/variant, the best finished checkpoint gets the curated name
    (see champion_name) and stays listed; every other checkpoint is unlisted (files and records are kept).
    "Best" is decided on validation only: final-policy mean, then learning-curve AUC, then fewer collapses.
    Idempotent; if a later run beats a champion, the name moves to it and the old one gets its run id back."""
    from backend.research.reporting import report

    best: dict[str, tuple] = {}
    for g in report.groups():
        name = champion_name(g["algo"], g["variant"])
        for r in g["runs"]:
            if r["model_id"] is None or r["final_mean"] is None or get(r["model_id"]) is None:
                continue
            key = (r["final_mean"], r["auc"] or 0.0, -r["degradations"], -r["run_id"])
            if name not in best or key > best[name][0]:
                best[name] = (key, r, g)
    out = []
    for name, (_, r, g) in sorted(best.items()):
        holder = get(name)
        if holder and holder["run_id"] != r["run_id"]:  # dethroned champion gives the name back
            rename(name, derived_id(db.one("SELECT * FROM runs WHERE id=?", (holder["run_id"],))))
        rename(r["model_id"], name)
        out.append({"name": name, "label": g["label"], "run_id": r["run_id"], "seed": r["seed"],
                    "total_steps": g["total_steps"], "final_mean": r["final_mean"], "auc": r["auc"],
                    "degradations": r["degradations"], "group_seeds": g["n_seeds"]})
    marks = ",".join("?" * len(out))
    with db.tx() as con:
        con.execute(f"UPDATE models SET listed = CASE WHEN id IN ({marks}) OR status IN ('promoted','training') "
                    f"THEN 1 ELSE 0 END", [c["name"] for c in out])
    return out


class Policy:
    """Greedy policy of a registered model. Diagnostics expose only what the model really computes."""

    def __init__(self, row: dict):
        if (row["obs_dim"], row["obs_version"], row["n_actions"]) != (OBS_DIM, OBS_VERSION, N_ACTIONS):
            raise IncompatibleModel(
                f"{row['id']} expects obs_dim={row['obs_dim']} obs_version={row['obs_version']} "
                f"n_actions={row['n_actions']}; this build has {OBS_DIM}/{OBS_VERSION}/{N_ACTIONS}")
        path = str(db.resolve(row["checkpoint_path"]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint missing for {row['id']}: {row['checkpoint_path']}")
        self.row, self.algo = row, row["algo"]
        if self.algo == "xdqn":
            from backend.rl.agents.dqn import DQNConfig
            from backend.rl.experimental import make_agent
            ck = torch.load(path, map_location="cpu", weights_only=True)
            hyper = ck["hyper"]
            agent = make_agent(ck["variant"], DQNConfig(**{**hyper, "hidden": tuple(hyper["hidden"]), "buffer_size": 1}), 1)
            agent.net.load_state_dict(ck["net"])
            self._net = agent.net.eval()
            self.diag_kind = "q_values"
        elif self.algo in ("dqn", "ppo"):
            from stable_baselines3 import DQN, PPO
            model = {"dqn": DQN, "ppo": PPO}[self.algo].load(path, device="cpu")
            model.policy.set_training_mode(False)
            if self.algo == "dqn":
                self._net, self.diag_kind = model.policy.q_net, "q_values"
            else:
                # Stateless forward pass. SB3's policy.get_distribution() mutates a distribution object shared
                # by the whole policy, so two threads using one cached model (a live match + a headless series)
                # could read each other's action probabilities. Never route inference through it.
                pol = model.policy
                from stable_baselines3.common.policies import BasePolicy

                def probs(t):
                    features = BasePolicy.extract_features(pol, t, pol.pi_features_extractor)
                    return torch.softmax(pol.action_net(pol.mlp_extractor.forward_actor(features)), dim=-1)

                self._net = probs
                self.diag_kind = "action_probs"
        else:
            raise IncompatibleModel(f"unknown algorithm {self.algo}")

    @torch.no_grad()
    def values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values (DQN family) or action probabilities (PPO), shape [B, 2]."""
        return self._net(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.values(obs).argmax(1)

    def act_diag(self, obs) -> tuple[int, dict]:
        v = self.values(np.asarray(obs, dtype=np.float32)[None])[0]
        return int(v.argmax()), {self.diag_kind: [float(x) for x in v]}


@functools.lru_cache(maxsize=16)
def _load(mid: str, path: str, mtime: float) -> Policy:
    return Policy(get(mid))


def load(mid: str) -> Policy:
    row = get(mid)
    if row is None:
        raise KeyError(f"unknown model {mid}")
    path = str(db.resolve(row["checkpoint_path"]))
    return _load(mid, path, os.path.getmtime(path) if os.path.exists(path) else 0.0)
