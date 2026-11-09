"""Proportional prioritized experience replay (Schaul et al. 2016). extra: per_alpha (0.6), per_beta (0.4 -> 1)."""
import numpy as np

from backend.rl.agents.dqn import DQNAgent


class PrioritizedReplay(DQNAgent):
    name = "per"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.alpha = float(self.cfg.extra.get("per_alpha", 0.6))
        self.beta0 = float(self.cfg.extra.get("per_beta", 0.4))
        self.prio = np.zeros(self.cfg.buffer_size)  # stored already raised to alpha
        self.max_prio = 1.0

    def on_store(self, index, env, done):
        super().on_store(index, env, done)
        self.prio[index] = self.max_prio

    def sample(self):
        # ponytail: O(buffer) cumsum per batch (~0.1 ms at 100k); switch to a sum-tree if buffers grow 10x
        cum = np.cumsum(self.prio[: self.size])
        idx = np.minimum(np.searchsorted(cum, self.rng.random(self.cfg.batch_size) * cum[-1]), self.size - 1)
        beta = min(1.0, self.beta0 + (1.0 - self.beta0) * self.step / self.total_steps)
        w = (self.size * self.prio[idx] / cum[-1]) ** -beta
        return idx, (w / w.max()).astype(np.float32)

    def after_update(self, idx, td_abs):
        super().after_update(idx, td_abs)
        p = (td_abs.cpu().numpy() + 1e-3) ** self.alpha
        self.prio[idx] = p
        self.max_prio = max(self.max_prio, float(p.max()))
