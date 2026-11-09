import random
from dataclasses import replace

import pytest

from backend.game.engine import DIFFICULTIES, GameConfig, World, replay

CFG = GameConfig()


def run(seed, policy, n=400, n_birds=1, cfg=CFG):
    w = World(cfg, seed, n_birds)
    trace = []
    for _ in range(n):
        if w.done:
            break
        w.step(policy(w))
        trace.append(tuple((b.y, b.vy, b.alive, b.score) for b in w.birds))
    return w, trace


def hover(w):  # crude controller, only used to exercise the engine (never presented as an agent)
    i = w.next_index()
    top, bottom = w.gap(i)
    return [b.y > (top + bottom) / 2 + 12 and b.vy >= 0 for b in w.birds]


def test_gravity_and_flap_physics():
    w = World(CFG, 0)
    dt = 1 / CFG.tick_rate
    w.step([0])
    assert w.birds[0].vy == pytest.approx(CFG.gravity * dt)
    assert w.birds[0].y == pytest.approx(CFG.start_y + CFG.gravity * dt * dt)
    w.step([1])
    assert w.birds[0].vy == -CFG.flap_velocity
    for _ in range(100):
        if w.done:
            break
        w.step([0])
    assert w.birds[0].vy <= CFG.max_fall_speed


def test_floor_and_ceiling_terminate():
    w, _ = run(0, lambda w: [0])
    assert w.done and w.birds[0].death == "floor" and w.birds[0].score == 0
    w, _ = run(0, lambda w: [1])
    assert w.done and w.birds[0].death == "ceiling"


def test_obstacle_collision_and_scoring():
    # A bird parked at the top of the world must hit the first upper block, not pass through it.
    cfg = replace(CFG, gravity=0.001, flap_velocity=0.001, start_y=40.0)
    w, _ = run(3, lambda w: [0], n=2000, cfg=cfg)
    b = w.birds[0]
    assert b.death == "obstacle" and b.score == 0
    # it died as its front edge reached the first obstacle
    assert w.obstacle_x(0) - w.scroll - cfg.bird_x == pytest.approx(cfg.bird_radius, abs=cfg.speed / cfg.tick_rate)
    # A competent controller clears obstacles and the score equals obstacles fully passed.
    w, _ = run(3, hover, n=3000)
    b = w.birds[0]
    assert b.score >= 5
    passed = sum(w.obstacle_x(i) + CFG.obstacle_width < CFG.bird_x + w.scroll - CFG.bird_radius for i in range(b.score + 3))
    assert b.score in (passed, passed - 1)  # -1 only if it died on the very tick it cleared one


def test_deterministic_under_same_seed_and_actions():
    rng = random.Random(7)
    actions = [rng.random() < 0.08 for _ in range(600)]
    it1, it2 = iter(actions), iter(actions)
    _, t1 = run(11, lambda w: [next(it1)], 600)
    _, t2 = run(11, lambda w: [next(it2)], 600)
    assert t1 == t2 and len(t1) > 5


def test_obstacle_field_depends_only_on_seed():
    a, b, c = World(CFG, 5), World(CFG, 5, n_birds=2), World(CFG, 6)
    b.step([1, 0])  # bird behaviour must not consume the obstacle RNG
    assert [a.gap(i) for i in range(50)] == [b.gap(i) for i in range(50)]
    assert [a.gap(i) for i in range(50)] != [c.gap(i) for i in range(50)]


def test_gaps_respect_config_bounds():
    for cfg in DIFFICULTIES.values():
        w = World(cfg, 1)
        prev = None
        for i in range(300):
            top, bottom = w.gap(i)
            assert cfg.gap_min - 1e-9 <= bottom - top <= cfg.gap_max + 1e-9
            assert top >= cfg.gap_margin - 1e-9 and bottom <= cfg.world_height - cfg.gap_margin + 1e-9
            center = (top + bottom) / 2
            if prev is not None:
                assert abs(center - prev) <= cfg.max_gap_shift + 1e-9
            prev = center


def test_shared_world_birds_are_independent_and_world_continues():
    # bird 0 never flaps (dies on the floor), bird 1 is steered and must keep going afterwards
    w = World(CFG, 3, n_birds=2)
    solo = World(CFG, 3, n_birds=1)
    for _ in range(1500):
        if w.done:
            break
        a1 = hover(w)[1]
        w.step([0, a1])
        if not solo.done:
            solo.step([hover(solo)[0]])
    b0, b1 = w.birds
    assert not b0.alive and b0.death == "floor" and b0.died_tick < 60
    assert b1.ticks > b0.ticks + 200 and b1.score >= 3  # the world was not reset or frozen
    # bird 1's trajectory is identical to playing alone on the same seed
    assert (b1.y, b1.vy, b1.score, b1.ticks) == (solo.birds[0].y, solo.birds[0].vy, solo.birds[0].score, solo.birds[0].ticks)


def test_replay_reproduces_a_recorded_game():
    w = World(CFG, 21, n_birds=2)
    flaps = [[], []]
    rng = random.Random(1)
    while not w.done and w.tick < 800:
        acts = [hover(w)[0], rng.random() < 0.07]
        w.step(acts)
        for i, a in enumerate(acts):
            if a:
                flaps[i].append(w.tick)
    r = replay(CFG, 21, flaps, 800)
    assert [(b.y, b.score, b.ticks, b.death) for b in r.birds] == [(b.y, b.score, b.ticks, b.death) for b in w.birds]


def test_snapshot_is_view_only_and_config_validation():
    w = World(CFG, 2)
    for _ in range(10):
        w.step([hover(w)[0]])
    s1 = w.snapshot()
    assert s1 == w.snapshot() and s1["tick"] == 10
    assert all(-CFG.obstacle_width <= o["x"] <= CFG.view_width for o in s1["obstacles"])
    with pytest.raises(ValueError):
        replace(CFG, obstacle_spacing=70.0)
