import os
import time

import numpy as np
import pytest

from backend.models import registry
from backend.rl.environments.env import OBS_DIM
from backend.rl.training import train
from backend.storage import db


def wait(run_id, statuses, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if run["status"] in statuses:
            return run
        time.sleep(0.3)
    raise AssertionError(f"run {run_id} still {run['status']} after {timeout}s: {run['error']}")


def test_each_algorithm_trains_and_registers_a_usable_model(lab):
    obs = np.random.default_rng(0).random((5, OBS_DIM), np.float32)
    for algo, mid in lab.items():
        row = registry.get(mid)
        assert row["status"] == "evaluated" and row["algo"] == algo and db.resolve(row["checkpoint_path"]).exists() and not os.path.isabs(row["checkpoint_path"])
        assert row["metrics"]["final_val"]["n"] == 50 and row["train_steps"] >= 3000
        run = db.one("SELECT * FROM runs WHERE id=?", (row["run_id"],))
        assert run["status"] == "completed" and run["code_version"] and run["hardware"]["packages"]["torch"]
        purposes = [e["purpose"] for e in db.query("SELECT purpose FROM evals WHERE run_id=?", (run["id"],))]
        assert purposes.count("periodic") >= 2 and purposes.count("final") == 1
        keys = {m["key"] for m in db.query("SELECT DISTINCT key FROM metrics WHERE run_id=?", (run["id"],))}
        assert {"sps", "rss_mb", "val_mean", "ep_score_mean", "loss"} <= keys
        policy = registry.load(mid)
        actions = policy.act(obs)
        assert actions.shape == (5,) and set(actions) <= {0, 1}
        action, diag = policy.act_diag(obs[0])
        assert list(diag) == ["action_probs" if algo == "ppo" else "q_values"] and action == actions[0]
        if algo == "ppo":
            assert abs(sum(diag["action_probs"]) - 1) < 1e-5


def test_incompatible_or_missing_checkpoints_are_refused(lab):
    row = registry.get(lab["xdqn"])
    with pytest.raises(registry.IncompatibleModel):
        registry.Policy({**row, "obs_dim": OBS_DIM + 1})
    with pytest.raises(FileNotFoundError):
        registry.Policy({**row, "checkpoint_path": row["checkpoint_path"] + ".missing"})
    with pytest.raises(ValueError):
        registry.set_status(lab["xdqn"], "promoted")  # promotion needs held-out evidence
    with pytest.raises(ValueError):
        registry.promote(lab["xdqn"], {})


def test_stop_checkpoints_and_resume_finishes_the_budget(lab):
    rid = train.create_run("xdqn", seed=3, total_steps=6000, eval_every=3000, eval_episodes=2,
                           checkpoint_every=2000, hyper={"learning_starts": 500})
    train.launch(rid)
    train.request_stop(rid)
    run = wait(rid, ("stopped", "completed", "failed"))
    assert run["status"] == "stopped" and 0 < run["steps"] < 6000, run["error"]
    ck = str(db.resolve(run["checkpoint_dir"]))
    assert os.path.exists(f"{ck}/last.pt") and os.path.exists(f"{ck}/state.json")
    assert not [f for f in os.listdir(ck) if f.endswith(".tmp")]  # atomic writes leave no partial files
    assert registry.get(run["model_id"])["status"] == "experimental"  # unfinished budget is not "evaluated"
    train.launch(rid, resume=True)
    run = wait(rid, ("completed", "failed", "stopped"))
    assert run["status"] == "completed" and run["steps"] == 6000, run["error"]
    assert registry.get(run["model_id"])["status"] == "evaluated"


def test_resource_limits_stop_runs_and_failures_are_recorded(lab):
    rid = train.create_run("xdqn", seed=4, total_steps=50_000, hyper={"learning_starts": 500},
                           limits={"max_memory_mb": 1})
    assert train.main(rid) == "limit"
    run = db.one("SELECT * FROM runs WHERE id=?", (rid,))
    assert run["steps"] <= 2000 and "memory limit" in run["error"]
    assert db.one("SELECT COUNT(*) AS n FROM events WHERE run_id=? AND level='warning'", (rid,))["n"] == 1
    rid = train.create_run("xdqn", seed=5, total_steps=50_000, limits={"max_wall_seconds": 0})
    assert train.main(rid) == "limit"
    rid = train.create_run("xdqn", seed=6, total_steps=5000, hyper={"learning_starts": 500, "batch_size": 0})
    assert train.main(rid) == "failed"  # empty batch -> non-finite loss -> recorded, not hidden
    run = db.one("SELECT * FROM runs WHERE id=?", (rid,))
    assert "non-finite" in run["error"]
    assert db.one("SELECT COUNT(*) AS n FROM events WHERE run_id=? AND level='error'", (rid,))["n"] == 1


def test_run_validation_and_identity_keys(lab):
    with pytest.raises(ValueError):
        train.create_run("sarsa")
    with pytest.raises(ValueError):
        train.create_run("ppo", variant="per")
    with pytest.raises(ValueError):
        train.create_run("xdqn", variant="nope")
    with pytest.raises(ValueError):
        train.create_run("dqn", hyper={"not_a_param": 1})
    a = db.one("SELECT * FROM runs WHERE id=?", (train.create_run("xdqn", seed=1, total_steps=1000),))
    b = db.one("SELECT * FROM runs WHERE id=?", (train.create_run("xdqn", seed=2, total_steps=1000),))
    c = db.one("SELECT * FROM runs WHERE id=?", (train.create_run("xdqn", "double", seed=1, total_steps=1000),))
    assert a["group_key"] == b["group_key"] != c["group_key"] and a["config_key"] != b["config_key"]
    assert a["kind"] == "baseline" and c["kind"] == "experimental"
    with db.tx() as con:
        con.execute("DELETE FROM runs WHERE id IN (?,?,?)", (a["id"], b["id"], c["id"]))


def test_device_benchmark_measures_instead_of_assuming(lab):
    out = train.benchmark_devices(0.3)
    assert out["chosen"] in out and out["cpu"] > 0
    assert train.resolve_device("auto") == out["chosen"]


def test_inference_is_thread_safe_and_matches_the_trained_policy(lab):
    """A cached policy is shared by live matches (event loop) and headless endpoints (thread pool)."""
    import threading

    import torch
    rng = np.random.default_rng(1)
    batches = [rng.random((1, OBS_DIM), np.float32) for _ in range(64)]
    for algo, mid in lab.items():
        policy = registry.load(mid)
        expected = [policy.values(b).copy() for b in batches]
        if algo == "ppo":  # the stateless path must equal SB3's own distribution
            sb3 = policy._net.__closure__  # noqa: SLF001 - sanity only
            assert sb3 is not None
            assert all(abs(float(e.sum()) - 1) < 1e-5 for e in expected)
        wrong = []

        def worker(k):
            for _ in range(200):
                for i in range(k, len(batches), 4):
                    if not np.array_equal(policy.values(batches[i]), expected[i]):
                        wrong.append((algo, i))

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not wrong, f"{algo}: {len(wrong)} cross-thread mismatches"
    # and the PPO probabilities are exactly the model's own (single-threaded reference through SB3)
    from stable_baselines3 import PPO
    row = registry.get(lab["ppo"])
    model = PPO.load(db.resolve(row["checkpoint_path"]), device="cpu")
    with torch.no_grad():
        ref = model.policy.get_distribution(torch.as_tensor(batches[0])).distribution.probs.numpy()
    assert np.allclose(registry.load(lab["ppo"]).values(batches[0]), ref, atol=1e-6)


def test_curation_keeps_one_named_champion_per_group_and_renames_everywhere(lab):
    from backend.api.game import Match, load_replay
    m = Match("ai_vs_ai", [lab["dqn"], lab["ppo"]], seed=3, limit_ticks=60)
    while not m.over:
        m.tick()
    match_id = m.save()["match_id"]
    before = {r["id"]: r["status"] for r in registry.list_models(include_unlisted=True)}
    try:
        champs = {c["name"]: c for c in registry.curate()}
        assert {"PPO_Baseline", "DQN_Baseline", "XDQN_Control"} <= set(champs)
        assert registry.champion_name("xdqn", "prior_explore") == "XDQN_PriorExplore"
        listed = registry.list_models()
        assert {r["id"] for r in listed if r["status"] != "training"} == set(champs)  # nothing else is listed
        assert len(registry.list_models(include_unlisted=True)) == len(before)       # nothing was deleted
        assert registry.get(lab["ppo"]) is None                                       # the id itself was renamed
        ppo = registry.get("PPO_Baseline")
        assert db.one("SELECT model_id FROM runs WHERE id=?", (ppo["run_id"],))["model_id"] == "PPO_Baseline"
        assert db.one("SELECT COUNT(*) AS n FROM evals WHERE model_id=?", (lab["ppo"],))["n"] == 0
        assert db.one("SELECT COUNT(*) AS n FROM evals WHERE model_id='PPO_Baseline'")["n"] >= 3
        stored = db.one("SELECT participants, result FROM matches WHERE id=?", (match_id,))
        assert [p["model_id"] for p in stored["participants"]] == ["DQN_Baseline", "PPO_Baseline"]
        assert [p["label"] for p in load_replay(match_id)["participants"]] == ["DQN_Baseline", "PPO_Baseline"]
        assert registry.load("PPO_Baseline").act(np.zeros((1, OBS_DIM), np.float32)).shape == (1,)  # still playable
        assert registry.model_id(db.one("SELECT * FROM runs WHERE id=?", (ppo["run_id"],))) == "PPO_Baseline"
        assert [c["name"] for c in registry.curate()] == sorted(champs)  # idempotent
    finally:  # other tests address models by their run-derived ids
        for row in registry.list_models(include_unlisted=True):
            run = db.one("SELECT * FROM runs WHERE id=?", (row["run_id"],))
            registry.rename(row["id"], registry.derived_id(run))
        with db.tx() as con:
            con.execute("UPDATE models SET listed=1")
    assert registry.get(lab["ppo"])["status"] == before[lab["ppo"]]


def test_shipped_seed_gives_a_fresh_clone_playable_models(tmp_path, monkeypatch):
    """What a new user gets on first start: the snapshot in seed/ becomes a working lab."""
    if not (db.SEED / "lab.db").exists():
        pytest.skip("no seed snapshot in this checkout")
    monkeypatch.setenv("FLAPLAB_HOME", str(tmp_path))
    monkeypatch.setenv("FLAPLAB_SEED", "1")
    db.init()
    listed = registry.list_models()
    assert len(listed) >= 2 and all(m["id"].split("_")[0] in ("PPO", "DQN", "XDQN") for m in listed)
    assert all(not os.path.isabs(m["checkpoint_path"]) and db.resolve(m["checkpoint_path"]).exists() for m in listed)
    best = registry.best_verified()
    assert best is not None and best["listed"] == 1
    from backend.rl.evaluation.evaluate import evaluate
    res = evaluate(registry.load(best["id"]).act, n_episodes=3, max_score=20)
    assert res["mean"] == 20  # the shipped best model really flies
    assert db.one("SELECT COUNT(*) AS n FROM matches")["n"] == 0 and db.one("SELECT COUNT(*) AS n FROM experiments")["n"] >= 1
    assert db.seed_if_empty() is False  # never overwrites an existing lab
