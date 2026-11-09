"""Gymnasium environment over the shared engine. Spec id: FlapLab-v1.

Observation (float32, OBS_DIM=12, see OBS_NAMES) and action semantics are documented in
docs/architecture.md. `observe()` is the ONLY observation function: training, evaluation
and the live game server all call it, so a policy sees identical inputs everywhere.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from backend.game.engine import ENGINE_VERSION, GameConfig, World

OBS_VERSION = "1"
ENV_VERSION = f"FlapLab-v1/engine{ENGINE_VERSION}/obs{OBS_VERSION}"
X_NORM = 432.0  # fixed constants (not config-derived) so scenario changes stay visible to the agent
SPEED_NORM = 200.0
OBS_NAMES = [
    "bird_y",            # y / world_height (0 ceiling, 1 floor)
    "bird_vy",           # vy / max_fall_speed (negative = rising)
    "next_left_dx",      # (next obstacle left edge - bird x) / 432
    "next_right_dx",     # (next obstacle right edge - bird x) / 432
    "next_gap_center",   # gap centre y / world_height
    "below_gap_top",     # (bird y - gap top) / world_height  (>0 = below the upper block edge)
    "above_gap_bottom",  # (gap bottom - bird y) / world_height (>0 = above the lower block edge)
    "next_gap_height",   # gap height / world_height
    "next2_left_dx",     # same as next_left_dx for the obstacle after next
    "next2_gap_center",
    "next2_gap_height",
    "world_speed",       # speed / 200
]
OBS_DIM = len(OBS_NAMES)
N_ACTIONS = 2  # 0 = do not flap, 1 = flap

# World-seed partitions. Training never draws below TRAIN_SEED_MIN, so validation/test fields are unseen.
VAL_SEED0, TEST_SEED0, TRAIN_SEED_MIN, TRAIN_SEED_MAX = 10_000, 20_000, 1_000_000, 2_000_000_000


def observe(world: World, bird) -> list[float]:
    c = world.cfg
    h = c.world_height
    i = world.next_index()
    top, bottom = world.gap(i)
    top2, bottom2 = world.gap(i + 1)
    left = world.obstacle_x(i) - world.scroll - c.bird_x
    return [
        bird.y / h,
        bird.vy / c.max_fall_speed,
        left / X_NORM,
        (left + c.obstacle_width) / X_NORM,
        (top + bottom) / (2 * h),
        (bird.y - top) / h,
        (bottom - bird.y) / h,
        (bottom - top) / h,
        (left + c.obstacle_spacing) / X_NORM,
        (top2 + bottom2) / (2 * h),
        (bottom2 - top2) / h,
        c.speed / SPEED_NORM,
    ]


@dataclass(frozen=True)
class RewardConfig:
    """Baseline reward. Evaluation never uses it: agents are judged on game score."""
    pass_obstacle: float = 1.0
    crash: float = -1.0
    survive: float = 0.0  # per tick
    flap: float = 0.0     # per flap (action cost), off by default

    def to_dict(self) -> dict:
        return asdict(self)


class FlapEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: GameConfig | None = None, reward: RewardConfig | None = None,
                 max_episode_steps: int = 5000, render_mode: str | None = None):
        self.cfg = config or GameConfig()
        self.rew = reward or RewardConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(-4.0, 4.0, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.world: World | None = None
        self._components = dict.fromkeys(("pass_obstacle", "crash", "survive", "flap"), 0.0)

    def _obs(self) -> np.ndarray:
        return np.array(observe(self.world, self.world.birds[0]), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        world_seed = (options or {}).get("world_seed")
        if world_seed is None:
            world_seed = int(self.np_random.integers(TRAIN_SEED_MIN, TRAIN_SEED_MAX))
        self.world = World(self.cfg, world_seed)
        self._components = dict.fromkeys(self._components, 0.0)
        return self._obs(), {"world_seed": world_seed}

    def step(self, action):
        world, bird, rew = self.world, self.world.birds[0], self.rew
        before = bird.score
        world.step((int(action),))
        parts = {
            "pass_obstacle": rew.pass_obstacle * (bird.score - before),
            "crash": 0.0 if bird.alive else rew.crash,
            "survive": rew.survive if bird.alive else 0.0,
            "flap": rew.flap if action else 0.0,
        }
        for k, v in parts.items():
            self._components[k] += v
        terminated = not bird.alive
        truncated = not terminated and world.tick >= self.max_episode_steps
        info = {"score": bird.score}
        if terminated or truncated:
            info["reward_components"] = dict(self._components)
            info["death"] = bird.death
            info["world_seed"] = world.seed
        return self._obs(), sum(parts.values()), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi" or self.world is None:
            return None
        b = self.world.birds[0]
        return f"tick={self.world.tick} y={b.y:.1f} vy={b.vy:.1f} score={b.score} alive={b.alive}"

    def close(self):
        self.world = None


gym.register(id="FlapLab-v1", entry_point="backend.rl.environments.env:FlapEnv")
