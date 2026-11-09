"""Crash-focused replay: half of every batch is drawn from transitions recorded within K steps before a crash,
the other half uniformly. No importance correction (a known bias, listed in the hypothesis failure modes).
Related work: prioritized replay (Schaul et al. 2016), episodic backward update (Lee et al. 2019). Novelty: UNVERIFIED.
extra: crash_window (15), crash_fraction (0.5)."""
from collections import deque

import numpy as np

from backend.rl.agents.dqn import DQNAgent


class CrashFocus(DQNAgent):
    name = "crash_focus"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.window = int(self.cfg.extra.get("crash_window", 15))
        self.near = np.zeros(self.cfg.buffer_size, bool)
        self.recent = [deque(maxlen=self.window) for _ in range(self.cfg.n_envs)]

    def on_store(self, index, env, done):
        super().on_store(index, env, done)
        self.near[index] = False  # slot reused
        self.recent[env].append(index)
        if done:
            self.near[list(self.recent[env])] = True
            self.recent[env].clear()

    def sample(self):
        idx, w = super().sample()
        pool = np.flatnonzero(self.near[: self.size])
        k = int(len(idx) * float(self.cfg.extra.get("crash_fraction", 0.5)))
        if len(pool) and k:
            idx = idx.copy()
            idx[:k] = self.rng.choice(pool, k)
        return idx, w
