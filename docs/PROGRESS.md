# Status

What works, what was measured, and what is still open. "Verified" means it was executed and
observed, with the test or procedure named. Last updated 2026-11-09.

## What works

| Area | State | Evidence |
|---|---|---|
| Deterministic simulation shared by every mode | verified | `tests/test_engine.py` |
| Human play (keyboard, mouse, touch) | verified | browser pass + `tests/test_api.py::test_mode_human` |
| Trained agent plays with its real policy and real diagnostics | verified | `test_mode_ai_uses_the_real_policy…` (streamed action equals the model's output) |
| Human vs AI in one world; the survivor keeps flying | verified | browser pass + `test_mode_human_vs_ai_shares_one_world…` |
| AI vs AI, multi-seed series, exact replays | verified | `test_mode_ai_vs_ai_and_exact_replay`; concurrent series are identical |
| DQN and PPO baselines train to usable checkpoints | verified | benchmark below |
| Training start / safe stop / resume from the dashboard | verified | browser pass + `test_training_can_be_launched_monitored_and_stopped` |
| Bounded research sessions (start, pause, resume, stop) | verified | two real sessions + `tests/test_research.py` |
| Failed experiments and resource-limit stops are recorded | verified | `test_failed_implementation_is_recorded_as_a_result`, `test_resource_limits_stop_runs…` |
| Everything survives a restart | verified | real restarts + `test_everything_survives_an_api_restart` |
| A fresh clone starts with the shipped models | verified | `test_shipped_seed_gives_a_fresh_clone_playable_models` |

Test suites: `python -m pytest -q` (56 backend tests) and `cd frontend && npm test` (118 tests).
Browser checks were scripted passes in Chromium at 1648x760, 1440x900 and 820x1180 (touch).

## Research log

All runs: 150,000 environment steps, validation score = obstacles cleared, capped at 200.

**Baselines** (5 training seeds each; final policy on 50 fixed validation worlds):

| Algorithm | Final mean [95% CI over seeds] | Learning-curve AUC | Collapses per run |
|---|---|---|---|
| PPO (Stable-Baselines3) | 188.3 [172.9, 199.2] | 0.67 | 0.0 |
| DQN (Stable-Baselines3) | 124.1 [48.9, 194.9] | 0.24 | 4.0 |
| xdqn (in-house DQN, control) | 123.9 [48.9, 197.4] | 0.33 | 3.4 |

Held-out test of each baseline's best model: 200 in-distribution for all three; mean over six
out-of-distribution scenarios: PPO 126.9, xdqn 126.1, DQN 99.5. Best verified model: `PPO_Baseline`.

**Experiments** (candidate minus control, 95% bootstrap CI over seeds):

| # | Hypothesis | Stage | Decision | AUC difference | Final-mean difference |
|---|---|---|---|---|---|
| 1 | H002 soft target updates | screening, 5 seeds | inconclusive | +0.009 [−0.117, +0.133] | +33 [−77, +141] |
| 2 | H007 crash-focused replay | screening | inconclusive | −0.049 [−0.187, +0.079] | +14 [−95, +116] |
| 3 | H001 Double DQN | screening | inconclusive | +0.071 [−0.048, +0.182] | +16 [−80, +115] |
| 4 | H004 collapse guard | screening | inconclusive | −0.104 [−0.229, +0.021] | −49 [−142, +53] |
| 5 | H005 survivor action prior | screening | promising | +0.122 [+0.007, +0.233] | +32 [−63, +129] |
| 6 | H005 survivor action prior | confirmation, 8 fresh seeds | confirmed (AUC only) | +0.167 [+0.103, +0.229] | +7.6 [−75, +88] |

What experiment 6 shows: drawing exploratory actions from the agent's own "survivor" flap frequency
makes the in-house DQN learn faster at the same compute (wall-clock ratio 0.99). What it does not
show: a better final policy, fewer collapses, or anything competitive with PPO (pooled AUC 0.44
over 13 seeds against PPO's 0.67). Novelty is unverified: no literature search has been run, and
nothing here is claimed to be a new algorithm.

Not yet tested: H003 n-step returns, H006 prioritized replay, H008 auxiliary dynamics loss,
H009 dueling heads. No ablations or hybrid variants have been run.

## The shipped models

`seed/` carries the lab as it stood after these experiments: 62 completed runs with their metrics
and evaluations, the experiment reports, and the policy file of every run. The model list shows the
best run of each algorithm/variant (by final validation mean, then AUC, then fewer collapses):

| Model | What it is | Run AUC | Collapses in its run |
|---|---|---|---|
| `PPO_Baseline` | standard PPO; the best verified model | 0.72 | 0 |
| `DQN_Baseline` | standard DQN | 0.42 | 2 |
| `XDQN_Control` | in-house DQN without modifications | 0.52 | 2 |
| `XDQN_PriorExplore` | survivor action prior (experiments 5–6) | 0.47 | 1 |
| `XDQN_Double` | Double DQN targets | 0.42 | 2 |
| `XDQN_Softtarget` | Polyak-averaged targets | 0.42 | 0 |
| `XDQN_CrashFocus` | crash-focused replay | 0.36 | 0 |
| `XDQN_CollapseGuard` | rollback on collapse | 0.36 | 5 |

All eight reach the validation cap of 200; they differ in how fast and how steadily they got there.
Every other checkpoint is unlisted but present (`GET /api/models?all=true`).

## Known limits

- One environment with a 12-value observation; results may not transfer.
- The score cap saturates strong agents, so differences show in learning speed and stability.
- Bootstrap intervals over 3–13 seeds are coarse; only the confirmation stage counts as evidence.
- Training is seeded but not bit-reproducible across machines.
- Developed and verified on Ubuntu 24.04 under WSL2. Native Windows and macOS are untested.
- No automated browser end-to-end suite; browser verification was a scripted manual pass.
- The optional hook that hands a research brief to a coding agent has only been run with a stub.
- `docs/api-fixtures/` are samples captured before the models were renamed; they still carry ids
  such as `ppo-r8` and paths from the capture machine.

## A bug worth knowing about

PPO inference originally went through Stable-Baselines3's `policy.get_distribution()`, which
mutates a distribution object shared by the whole policy. A live match and a headless series using
the same cached model from two threads could read each other's action probabilities and record a
wrong crash. Inference is now a stateless forward pass (`backend/models/registry.py`), with a
regression test that fails on the old code
(`test_inference_is_thread_safe_and_matches_the_trained_policy`). Training and every evaluation run
in single-threaded processes and were never affected.

## How it was built

Backend, research system and integration tests were written with Claude Code. The whole frontend
was written by Codex. The two worked through a small open bridge,
[claude-codex-bridge](https://github.com/jalv92/claude-codex-bridge): one side publishes a task
with a contract and real fixtures, the other implements, and each round is verified in a browser
before the next. See [`codex-integration.md`](codex-integration.md).
