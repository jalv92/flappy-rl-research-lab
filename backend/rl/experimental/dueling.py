"""Dueling value/advantage heads (Wang et al. 2016)."""
import torch.nn as nn

from backend.rl.agents.dqn import DQNAgent, mlp
from backend.rl.environments.env import N_ACTIONS, OBS_DIM


class DuelingNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.trunk = nn.Sequential(*list(mlp((OBS_DIM, *hidden), 1).children())[:-1])
        self.value, self.adv = nn.Linear(hidden[-1], 1), nn.Linear(hidden[-1], N_ACTIONS)

    def forward(self, x):
        h = self.trunk(x)
        a = self.adv(h)
        return self.value(h) + a - a.mean(1, keepdim=True)


class Dueling(DQNAgent):
    name = "dueling"

    def build_net(self):
        return DuelingNet(self.cfg.hidden)
