# Architecture

```
browser (frontend/, Codex)            FastAPI process (backend/api)                subprocesses
┌──────────────────────────┐  WS /ws/game   ┌───────────────────────────┐   ┌───────────────────────────────┐
│ renderer + input         │◄──────────────►│ game sessions (asyncio)   │   │ trainer  (one per run)        │
│ lab dashboard (polling)  │  REST /api/*   │  └ Match ─► World (engine)│   │  python -m backend.rl.training│
└──────────────────────────┘◄──────────────►│ REST: models, training,   │──►│  .train RUN_ID                │
                                            │ research, compare, status │   │ research controller (one)     │
                                            └────────────┬──────────────┘   │  python -m backend.research   │
                                                         │ sqlite (WAL)     │  .orchestration.controller ID │
                                                experiments/lab.db ◄────────┴───────────────────────────────┘
                                                checkpoints/run_XXXXX/, experiments/{replays,reports,logs}/
```

## One simulation

`backend/game/engine.py` (`World`) is the only physics implementation. It has no renderer, no wall
clock and no global RNG: the state after N ticks is a pure function of `(GameConfig, seed, actions)`.

- Fixed timestep: 30 ticks/s. Semi-implicit Euler; a flap *sets* the vertical velocity.
- Obstacles are generated lazily from a per-world `random.Random(seed)` that is consumed only by
  obstacle generation, in index order. The field therefore never depends on what any bird does,
  which is what makes Human-vs-AI and AI-vs-AI fair: N birds share one `World`, with independent
  position, velocity, score and death, and never collide with each other.
- A bird dies on the floor, the ceiling (no flying over obstacles) or an obstacle block
  (circle-vs-rectangle test). The world keeps stepping while any bird is alive.
- `score` = obstacles fully cleared; `distance` = ticks survived × scroll per tick.
- `ENGINE_VERSION` must be bumped on any trajectory-changing edit; replays and models record it.

Who steps it:

| Use | Stepper |
|---|---|
| Human / AI / Human-vs-AI / AI-vs-AI / replay | `backend/api/game.py` `Match.tick()`, paced by the WebSocket session |
| Headless AI-vs-AI series | `game.run_series()` (same `Match`, no pacing) |
| Training | `FlapEnv.step()` (`backend/rl/environments/env.py`) |
| Evaluation | `evaluate.play()` (lock-step batch of worlds) |

### Why the server is authoritative

Running the engine in the browser would need a second (TypeScript) physics implementation, and the
spec forbids training and gameplay physics from diverging. Instead the browser is a renderer: the
server streams a snapshot per tick over a local WebSocket and receives `flap` events. On localhost
the cost is at most one tick (33 ms) of input delay, which the client masks with instant local
feedback and inter-tick interpolation. AI inference runs server-side with the real PyTorch policy on
the authoritative state. Render speed cannot influence outcomes, and every match is exactly
replayable from `seed + flap ticks`.

## Environment specification: `FlapLab-v1`

- Action space `Discrete(2)`: `0` do not flap, `1` flap. One action per tick.
- Observation `Box(-4, 4, (12,), float32)`, produced by `env.observe()` (the single observation
  function, also used by the live game server):

| # | name | definition |
|---|---|---|
| 0 | bird_y | y / world_height (0 ceiling, 1 floor) |
| 1 | bird_vy | vy / max_fall_speed (negative = rising) |
| 2 | next_left_dx | (next obstacle left edge − bird x) / 432 |
| 3 | next_right_dx | (next obstacle right edge − bird x) / 432 |
| 4 | next_gap_center | gap centre y / world_height |
| 5 | below_gap_top | (bird y − gap top) / world_height |
| 6 | above_gap_bottom | (gap bottom − bird y) / world_height |
| 7 | next_gap_height | gap height / world_height |
| 8 | next2_left_dx | same as 2 for the obstacle after next |
| 9 | next2_gap_center | |
| 10 | next2_gap_height | |
| 11 | world_speed | speed / 200 |

  "Next" = first obstacle whose right edge is not yet behind the bird's back edge.
- Reward (baseline, `RewardConfig`): `+1` per obstacle cleared, `−1` on crash, `0` per tick,
  `0` per flap. All four are configurable and their per-episode sums are returned in
  `info["reward_components"]`. Survival and flap terms are off by default: no shaping has been
  justified experimentally. Agents are never judged on reward, only on game score.
- Termination = crash. Truncation = 5000 ticks (bootstrapped correctly, not treated as a crash).
- World seeds are partitioned: training ≥ 1,000,000; validation 10,000+; test 20,000+;
  AI-vs-AI series 30,000+ by default. Training can never see a validation or test field.
- `ENV_VERSION = "FlapLab-v1/engine1/obs1"` is stored on every run, model and evaluation, and model
  loading checks observation size/version and action count before use.

## Algorithms

- `dqn`, `ppo`: Stable-Baselines3, the established baselines (`backend/rl/baselines` is just SB3;
  the wiring is in `train.train_sb3`).
- `xdqn`: an in-house DQN (`backend/rl/agents/dqn.py`) with the same hyperparameters as the SB3 DQN.
  Experimental variants (`backend/rl/experimental/<name>.py`) each override one hook; `a+b` composes
  them for hybrids and ablations. `xdqn` vs `dqn` is an implementation check; `xdqn+variant` vs
  `xdqn` isolates one change.

Networks are small MLPs (128×128, PPO 64×64). `python -m scripts.lab bench` measures update
throughput on CPU and CUDA and `device="auto"` uses the faster one. On the development machine CPU
wins by ~2.4× for these sizes, so trainers run single-threaded on CPU and scale by running several
processes; the GPU is used only when a run asks for `device="cuda"` (capped by `max_gpu_mem_mb`).

## Training runs

One subprocess per run, fully described by its `runs` row. The trainer logs metrics every 1000
steps, evaluates on fixed validation worlds, writes atomic checkpoints (`*.tmp` → `os.replace`),
checks the stop flag and its limits (wall-clock, RSS, GPU fraction), and registers the model.
`stopped`, `limit` and `interrupted` runs resume from `last.*` + `state.json`. A run that crashes is
recorded as `failed` with its traceback; nothing is swallowed.

## Storage

SQLite (`experiments/lab.db`, WAL) holds runs, metrics, evaluations, models, hypotheses,
experiments, sessions, matches, events. Files hold checkpoints, replays, experiment reports and
trainer logs. `FLAPLAB_HOME` relocates both directories (tests use a temp dir). Both are gitignored:
they are machine-local and reproducible from configs + seeds.

## Reproducibility and its limits

- Engine, obstacle fields, evaluation and replays are bit-deterministic on a given platform.
- Training is seeded (PyTorch, NumPy, env seed streams) but **not guaranteed bit-identical** across
  machines, BLAS builds or PyTorch versions, and not at all on CUDA (non-deterministic kernels).
  Conclusions therefore rest on multiple seeds, never on re-obtaining one exact curve.
- Resuming a run restores weights, optimiser and replay buffer but re-seeds the environment
  streams, so a resumed run is statistically, not bitwise, equivalent to an uninterrupted one.
- Every run records git revision (+dirty flag), variant source hash, full config, seed, hardware and
  package versions.
