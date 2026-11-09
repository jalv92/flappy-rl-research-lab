"""Auxiliary forward-dynamics loss: a head predicts the scaled observation change from (features, action).
Related work: UNREAL auxiliary tasks (Jaderberg et al. 2017), SPR (Schwarzer et al. 2021).
extra: aux_weight (0.5), aux_scale (20)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.rl.agents.dqn import DQNAgent, mlp
from backend.rl.environments.env import N_ACTIONS, OBS_DIM


class AuxNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.trunk = nn.Sequential(*list(mlp((OBS_DIM, *hidden), 1).children())[:-1])
        self.q = nn.Linear(hidden[-1], N_ACTIONS)
        self.dyn = mlp((hidden[-1] + N_ACTIONS, 64), OBS_DIM)

    def forward(self, x):
        return self.q(self.trunk(x))


class AuxDynamics(DQNAgent):
    name = "aux_dynamics"

    def build_net(self):
        return AuxNet(self.cfg.hidden)

    def loss(self, b, weights):
        loss, td, extra = super().loss(b, weights)
        feat = self.net.trunk(b["obs"])
        pred = self.net.dyn(torch.cat([feat, F.one_hot(b["action"], N_ACTIONS).float()], 1))
        target = (b["next_obs"] - b["obs"]) * float(self.cfg.extra.get("aux_scale", 20.0))
        aux = (F.mse_loss(pred, target, reduction="none").mean(1) * (1.0 - b["done"])).mean()
        return loss + float(self.cfg.extra.get("aux_weight", 0.5)) * aux, td, {**extra, "aux_loss": aux.item()}
