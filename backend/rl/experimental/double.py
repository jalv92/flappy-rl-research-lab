"""Double DQN target (van Hasselt, Guez & Silver 2016): the online net picks the action, the target net scores it."""
import torch

from backend.rl.agents.dqn import DQNAgent


class DoubleDQN(DQNAgent):
    name = "double"

    def targets(self, b):
        with torch.no_grad():
            best = self.net(b["next_obs"]).argmax(1, keepdim=True)
            nxt = self.target(b["next_obs"]).gather(1, best).squeeze(1)
            return b["reward"] + b["gamma"] * (1.0 - b["done"]) * nxt
