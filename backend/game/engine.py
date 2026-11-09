"""Deterministic flap-through-gaps simulation.

The single physics implementation of the project: human play, AI play, competitions,
headless training and evaluation all step this class. It has no renderer, no wall clock
and no global RNG; the state after N ticks is a pure function of (config, seed, actions).

Coordinates: y grows downward, 0 = ceiling, world_height = floor. Birds never move
horizontally; the world scrolls past them at `speed`. Obstacles live at fixed world x.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace

ENGINE_VERSION = "1"  # bump on ANY change that alters trajectories


@dataclass(frozen=True)
class GameConfig:
    tick_rate: int = 30  # fixed simulation timestep = 1 / tick_rate seconds
    world_height: float = 512.0
    view_width: float = 432.0  # renderer hint only; never affects physics
    bird_x: float = 96.0
    bird_radius: float = 12.0  # circular hitbox
    start_y: float = 256.0
    gravity: float = 1200.0  # px/s^2
    flap_velocity: float = 360.0  # px/s upward, replaces current velocity
    max_fall_speed: float = 540.0
    speed: float = 140.0  # px/s horizontal scroll
    obstacle_width: float = 60.0
    obstacle_spacing: float = 210.0  # left edge to left edge
    first_obstacle_x: float = 420.0
    gap_min: float = 125.0
    gap_max: float = 165.0
    gap_margin: float = 50.0  # min distance from a gap edge to floor/ceiling
    max_gap_shift: float = 190.0  # max centre change between consecutive gaps (keeps it fair)

    def __post_init__(self):
        if self.obstacle_spacing <= self.obstacle_width + 2 * self.bird_radius:
            raise ValueError("obstacle_spacing must exceed obstacle_width + bird diameter")
        if self.gap_min <= 2 * self.bird_radius or self.gap_max < self.gap_min:
            raise ValueError("gap range must fit the bird")
        if self.gap_max + 2 * self.gap_margin > self.world_height:
            raise ValueError("gap_max + margins exceed world_height")
        if self.tick_rate <= 0 or self.speed <= 0:
            raise ValueError("tick_rate and speed must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


DIFFICULTIES: dict[str, GameConfig] = {
    "easy": replace(GameConfig(), gap_min=150.0, gap_max=185.0, speed=125.0, obstacle_spacing=230.0),
    "normal": GameConfig(),
    "hard": replace(GameConfig(), gap_min=110.0, gap_max=140.0, speed=160.0, obstacle_spacing=195.0),
}


class Bird:
    __slots__ = ("y", "vy", "alive", "score", "ticks", "flaps", "died_tick", "death")

    def __init__(self, y: float):
        self.y = y
        self.vy = 0.0
        self.alive = True
        self.score = 0  # obstacles fully cleared while alive
        self.ticks = 0  # ticks survived
        self.flaps = 0
        self.died_tick: int | None = None
        self.death: str | None = None  # "floor" | "ceiling" | "obstacle"


class World:
    """One obstacle field shared by `n_birds` independent birds (they never collide with each other)."""

    def __init__(self, config: GameConfig = GameConfig(), seed: int = 0, n_birds: int = 1):
        self.cfg = config
        self.seed = seed
        self.tick = 0
        self.birds = [Bird(config.start_y) for _ in range(n_birds)]
        self._rng = random.Random(seed)  # consumed ONLY by obstacle generation, in index order
        self._gaps: list[tuple[float, float]] = []  # (gap_top, gap_bottom) by obstacle index
        self._last_center = config.world_height / 2
        self._dt = 1.0 / config.tick_rate
        self._dx = config.speed / config.tick_rate

    # --- obstacle field (independent of bird state, hence identical for every participant) ---
    def gap(self, i: int) -> tuple[float, float]:
        c = self.cfg
        while len(self._gaps) <= i:
            h = self._rng.uniform(c.gap_min, c.gap_max)
            lo = max(c.gap_margin + h / 2, self._last_center - c.max_gap_shift)
            hi = min(c.world_height - c.gap_margin - h / 2, self._last_center + c.max_gap_shift)
            center = self._rng.uniform(lo, hi)
            self._last_center = center
            self._gaps.append((center - h / 2, center + h / 2))
        return self._gaps[i]

    def obstacle_x(self, i: int) -> float:
        """World x of the left edge of obstacle i."""
        return self.cfg.first_obstacle_x + i * self.cfg.obstacle_spacing

    @property
    def scroll(self) -> float:
        return self.tick * self._dx  # multiplication, not accumulation: no float drift

    def next_index(self) -> int:
        """Index of the first obstacle whose right edge is not yet behind the bird's back edge."""
        c = self.cfg
        q = (c.bird_x + self.scroll - c.bird_radius - c.first_obstacle_x - c.obstacle_width) / c.obstacle_spacing
        return max(0, math.ceil(q))

    @property
    def done(self) -> bool:
        return not any(b.alive for b in self.birds)

    # --- dynamics ---
    def step(self, actions) -> None:
        """Advance one fixed tick. actions[i] truthy = bird i flaps. Dead birds are ignored."""
        c = self.cfg
        dt = self._dt
        self.tick += 1
        nxt = self.next_index()
        left = self.obstacle_x(nxt) - self.scroll  # screen-space left edge
        top, bottom = self.gap(nxt)
        r = c.bird_radius
        # horizontal distance from bird centre to the obstacle column (0 when inside it)
        dxs = c.bird_x - min(max(c.bird_x, left), left + c.obstacle_width)
        dx2 = dxs * dxs
        r2 = r * r
        for bird, flap in zip(self.birds, actions):
            if not bird.alive:
                continue
            if flap:
                bird.vy = -c.flap_velocity
                bird.flaps += 1
            else:
                bird.vy = min(bird.vy + c.gravity * dt, c.max_fall_speed)
            y = bird.y = bird.y + bird.vy * dt
            bird.ticks += 1
            death = None
            if y + r >= c.world_height:
                death = "floor"
            elif y - r <= 0.0:
                death = "ceiling"
            elif dx2 < r2:
                dy = max(0.0, y - top)  # vertical distance to the upper block (occupies y <= gap_top)
                if dx2 + dy * dy < r2:
                    death = "obstacle"
                else:
                    dy = max(0.0, bottom - y)  # lower block occupies y >= gap_bottom
                    if dx2 + dy * dy < r2:
                        death = "obstacle"
            if death:
                bird.alive = False
                bird.death = death
                bird.died_tick = self.tick
            else:
                bird.score = nxt

    def distance(self, bird: Bird) -> float:
        return bird.ticks * self._dx

    # --- read-only view for renderers / API ---
    def snapshot(self) -> dict:
        c = self.cfg
        s = self.scroll
        i0 = max(0, math.ceil((s - c.first_obstacle_x - c.obstacle_width) / c.obstacle_spacing))
        i1 = math.floor((s + c.view_width - c.first_obstacle_x) / c.obstacle_spacing)
        obstacles = []
        for i in range(i0, i1 + 1):
            top, bottom = self.gap(i)
            obstacles.append({"i": i, "x": self.obstacle_x(i) - s, "gap_top": top, "gap_bottom": bottom})
        return {
            "tick": self.tick,
            "scroll_x": s,
            "obstacles": obstacles,
            "birds": [
                {"y": b.y, "vy": b.vy, "alive": b.alive, "score": b.score, "ticks": b.ticks,
                 "distance": self.distance(b), "flaps": b.flaps, "died_tick": b.died_tick, "death": b.death}
                for b in self.birds
            ],
        }


def replay(config: GameConfig, seed: int, flap_ticks: list[list[int]], max_ticks: int) -> World:
    """Re-simulate a recorded game. flap_ticks[i] = ticks (1-based, as World.tick after the step) where bird i flapped."""
    world = World(config, seed, n_birds=len(flap_ticks))
    sets = [set(t) for t in flap_ticks]
    while not world.done and world.tick < max_ticks:
        t = world.tick + 1
        world.step([t in s for s in sets])
    return world
