"""Compact DQN: the in-house control implementation that experimental variants modify.

Hyperparameters mirror the Stable-Baselines3 DQN baseline (backend/rl/baselines/sb3.py), so
"xdqn" vs SB3 "dqn" is an implementation sanity check, and "xdqn" vs "xdqn+variant" isolates
exactly one algorithmic change. Variants subclass DQNAgent and override the hook methods
(build_net / epsilon / explore / store / on_store / sample / targets / loss / after_update /
sync_target / on_episode). Names joined with '+' compose variants (see backend.rl.experimental).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.rl.environments.env import N_ACTIONS, OBS_DIM


@dataclass
class DQNConfig:
    n_envs: int = 8
    buffer_size: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 5e-4
    learning_starts: int = 2_000
    train_freq: int = 4          # env steps per gradient step
    target_update: int = 1_000   # env steps between hard target syncs
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_fraction: float = 0.2    # fraction of total steps over which epsilon is annealed
    hidden: tuple = (128, 128)
    max_grad_norm: float = 10.0
    extra: dict = field(default_factory=dict)  # variant-specific knobs, recorded with the run

    def to_dict(self) -> dict:
        return asdict(self)


def mlp(sizes, out) -> nn.Sequential:
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, out))


class DQNAgent:
    name = "control"

    def __init__(self, cfg: DQNConfig, total_steps: int, device: str = "cpu", seed: int = 0):
        self.cfg, self.total_steps, self.device = cfg, total_steps, torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = self.build_net().to(self.device)
        self.target = self.build_net().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        n = cfg.buffer_size
        self.buf = {"obs": np.zeros((n, OBS_DIM), np.float32), "next_obs": np.zeros((n, OBS_DIM), np.float32),
                    "action": np.zeros(n, np.int64), "reward": np.zeros(n, np.float32),
                    "done": np.zeros(n, np.float32), "gamma": np.full(n, cfg.gamma, np.float32)}
        self.pos = self.size = self.updates = self.step = 0
        self.last_target_sync = 0

    # ---------- hooks ----------
    def build_net(self) -> nn.Module:
        return mlp((OBS_DIM, *self.cfg.hidden), N_ACTIONS)

    def epsilon(self, step: int) -> float:
        c = self.cfg
        frac = min(1.0, step / max(1.0, c.eps_fraction * self.total_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def explore(self, n: int) -> np.ndarray:
        """Random actions used by epsilon-greedy (uniform in the control)."""
        return self.rng.integers(0, N_ACTIONS, n)

    def store(self, obs, action, reward, next_obs, done, ended) -> None:
        """Add one transition per env. `done` = terminated only (truncation still bootstraps);
        `ended` = terminated or truncated (episode boundary, used by multi-step variants)."""
        for i in range(len(obs)):
            self.add(obs[i], action[i], reward[i], next_obs[i], done[i], self.cfg.gamma, env=i)

    def add(self, obs, action, reward, next_obs, done, gamma, env: int = 0) -> int:
        p, buf = self.pos, self.buf
        buf["obs"][p], buf["next_obs"][p] = obs, next_obs
        buf["action"][p], buf["reward"][p], buf["done"][p], buf["gamma"][p] = action, reward, done, gamma
        self.pos = (p + 1) % self.cfg.buffer_size
        self.size = min(self.size + 1, self.cfg.buffer_size)
        self.on_store(p, env, bool(done))
        return p

    def on_store(self, index: int, env: int, done: bool) -> None:
        pass

    def sample(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (indices, importance weights or None)."""
        return self.rng.integers(0, self.size, self.cfg.batch_size), None

    def targets(self, b) -> torch.Tensor:
        with torch.no_grad():
            nxt = self.target(b["next_obs"]).max(1).values
            return b["reward"] + b["gamma"] * (1.0 - b["done"]) * nxt

    def loss(self, b, weights) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return (scalar loss, per-sample |TD error|, extra metrics)."""
        q = self.net(b["obs"]).gather(1, b["action"].unsqueeze(1)).squeeze(1)
        target = self.targets(b)
        per = F.smooth_l1_loss(q, target, reduction="none")
        loss = (per * weights).mean() if weights is not None else per.mean()
        return loss, (target - q).detach().abs(), {"q_mean": q.mean().item()}

    def after_update(self, idx: np.ndarray, td_abs: torch.Tensor) -> None:
        pass

    def sync_target(self, step: int) -> None:
        if step - self.last_target_sync >= self.cfg.target_update:
            self.target.load_state_dict(self.net.state_dict())
            self.last_target_sync = step

    def on_episode(self, score: int, length: int) -> None:
        pass

    def extra_metrics(self) -> dict:
        return {}

    # ---------- fixed machinery ----------
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.net(torch.as_tensor(obs, device=self.device)).cpu().numpy()

    def greedy(self, obs: np.ndarray) -> np.ndarray:
        return self.q_values(obs).argmax(1)

    def act(self, obs: np.ndarray, step: int) -> np.ndarray:
        actions = self.greedy(obs)
        mask = self.rng.random(len(obs)) < self.epsilon(step)
        if mask.any():
            actions[mask] = self.explore(int(mask.sum()))
        return actions

    def update(self, step: int) -> dict:
        self.step = step
        idx, w = self.sample()
        b = {k: torch.as_tensor(v[idx], device=self.device) for k, v in self.buf.items()}
        wt = torch.as_tensor(w, dtype=torch.float32, device=self.device) if w is not None else None
        loss, td_abs, extra = self.loss(b, wt)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.after_update(idx, td_abs)
        self.updates += 1
        self.sync_target(step)
        return {"loss": loss.item(), "td_abs": td_abs.mean().item(), **extra}

    def state(self) -> dict:
        """Policy + optimiser state (the replay buffer is saved separately, see save_buffer)."""
        return {"net": self.net.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                "updates": self.updates, "last_target_sync": self.last_target_sync, "extra": self.extra_state()}

    def load_state(self, s: dict) -> None:
        self.net.load_state_dict(s["net"])
        self.target.load_state_dict(s["target"])
        self.opt.load_state_dict(s["opt"])
        self.updates, self.last_target_sync = s["updates"], s["last_target_sync"]
        self.load_extra_state(s.get("extra") or {})

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, s: dict) -> None:
        pass

    def buffer_arrays(self) -> dict:
        return {**{k: v[: self.size] for k, v in self.buf.items()}, "pos": np.array(self.pos)}

    def load_buffer(self, arrays) -> None:
        self.size = len(arrays["obs"])
        for k in self.buf:
            self.buf[k][: self.size] = arrays[k]
        self.pos = int(arrays["pos"])
        for i in range(self.size):
            self.on_store(i, 0, bool(self.buf["done"][i]))
