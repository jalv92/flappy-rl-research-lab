import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from backend.rl.environments.env import (OBS_DIM, TRAIN_SEED_MIN, FlapEnv, RewardConfig)
from backend.rl.evaluation.evaluate import evaluate, play, SCENARIOS


def test_gymnasium_api_compliance():
    env = gym.make("FlapLab-v1")
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert info["world_seed"] >= TRAIN_SEED_MIN  # training never touches validation/test seeds
    env.close()


def test_reset_seed_reproduces_episode_stream():
    def rollout():
        env = FlapEnv()
        env.reset(seed=123)
        out = []
        for _ in range(3):
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(int(env.np_random.random() < 0.08))
                out.append((obs.tobytes(), r))
                done = term or trunc
            env.reset()
        return out
    assert rollout() == rollout()


def test_reward_components_are_configurable_and_logged():
    env = FlapEnv(reward=RewardConfig(pass_obstacle=1.0, crash=-2.0, survive=0.1, flap=-0.05))
    env.reset(seed=1)
    total, done, n, flaps = 0.0, False, 0, 0
    while not done:
        a = int(n % 9 == 0)
        _, r, term, trunc, info = env.step(a)
        total += r
        n += 1
        flaps += a
        done = term or trunc
    c = info["reward_components"]
    assert term and c["crash"] == -2.0
    assert abs(c["survive"] - 0.1 * (n - 1)) < 1e-9 and abs(c["flap"] + 0.05 * flaps) < 1e-9
    assert c["pass_obstacle"] == info["score"]
    assert abs(sum(c.values()) - total) < 1e-9


def test_truncation_is_not_termination():
    env = FlapEnv(max_episode_steps=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, term, trunc, _ = env.step(int(i == 3))
    assert trunc and not term


def test_evaluation_is_deterministic_and_splits_are_disjoint():
    def act(obs):  # flap when below the gap centre and falling
        return (obs[:, 0] > obs[:, 4] + 0.02) & (obs[:, 1] >= 0)
    a = evaluate(act, n_episodes=4, max_score=20)
    b = evaluate(act, n_episodes=4, max_score=20)
    t = evaluate(act, split="test", n_episodes=4, max_score=20)
    assert a["scores"] == b["scores"] and a["seed0"] != t["seed0"]
    assert max(a["scores"]) <= 20 and a["mean"] > 3
    assert set(a["success"]) == {"10", "25", "50", "100"}
    assert play(act, SCENARIOS["fast"], [1, 2], 5) == play(act, SCENARIOS["fast"], [1, 2], 5)
