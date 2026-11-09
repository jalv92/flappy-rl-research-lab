import os
import tempfile

import pytest

# Never touch the real lab database/checkpoints from tests.
os.environ["FLAPLAB_HOME"] = tempfile.mkdtemp(prefix="flaplab-test-")
os.environ["FLAPLAB_SEED"] = "0"  # start from an empty lab, not from the shipped snapshot
os.environ["FLAPLAB_AUTO_CURATE"] = "0"  # tests address models by run-derived ids; curation is tested explicitly


@pytest.fixture(scope="session")
def lab():
    """A tiny but REAL lab: one short training run per algorithm, through the production trainer."""
    from backend.rl.training import train
    from backend.storage import db

    db.init()
    models = {}
    for algo in ("xdqn", "dqn", "ppo"):
        rid = train.create_run(algo, seed=0, total_steps=3000, eval_every=1500, eval_episodes=2,
                               checkpoint_every=1500, hyper=None if algo == "ppo" else {"learning_starts": 500})
        assert train.main(rid) == "completed", db.one("SELECT error FROM runs WHERE id=?", (rid,))
        models[algo] = db.one("SELECT model_id FROM runs WHERE id=?", (rid,))["model_id"]
    return models
