"""Polyak-averaged target network instead of periodic hard copies (Lillicrap et al. 2016). extra: tau (0.005)."""
import torch

from backend.rl.agents.dqn import DQNAgent


class SoftTarget(DQNAgent):
    name = "softtarget"

    def sync_target(self, step):
        tau = float(self.cfg.extra.get("tau", 0.005))
        with torch.no_grad():
            for t, s in zip(self.target.parameters(), self.net.parameters()):
                t.lerp_(s, tau)
