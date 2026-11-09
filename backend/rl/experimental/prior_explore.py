"""Survivor action prior: epsilon-greedy's random actions are drawn from the action frequencies observed in
the agent's own longer-than-median episodes instead of uniformly. Starts uniform (identical to the control).
Closest prior work we know of: action priors / temporally-extended exploration (ez-greedy, Dabney et al. 2021).
Novelty: UNVERIFIED. extra: prior_rate (0.1), prior_min (0.02)."""
from collections import deque

import numpy as np

from backend.rl.agents.dqn import DQNAgent


class PriorExplore(DQNAgent):
    name = "prior_explore"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.p_flap = 0.5
        self.flaps = np.zeros(self.cfg.n_envs)
        self.steps = np.zeros(self.cfg.n_envs)
        self.lengths = deque(maxlen=200)

    def explore(self, n):
        return (self.rng.random(n) < self.p_flap).astype(np.int64)

    def store(self, obs, action, reward, next_obs, done, ended):
        super().store(obs, action, reward, next_obs, done, ended)
        self.flaps += action
        self.steps += 1
        lo, rate = float(self.cfg.extra.get("prior_min", 0.02)), float(self.cfg.extra.get("prior_rate", 0.1))
        for i in np.flatnonzero(ended):
            if self.lengths and self.steps[i] >= np.median(self.lengths):
                self.p_flap = float(np.clip((1 - rate) * self.p_flap + rate * self.flaps[i] / self.steps[i], lo, 0.5))
            self.lengths.append(self.steps[i])
            self.flaps[i] = self.steps[i] = 0

    def extra_metrics(self):
        return {**super().extra_metrics(), "p_flap": self.p_flap}

    def extra_state(self):
        return {**super().extra_state(), "p_flap": self.p_flap}

    def load_extra_state(self, s):
        super().load_extra_state(s)
        self.p_flap = s.get("p_flap", 0.5)
