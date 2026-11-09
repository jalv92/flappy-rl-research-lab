# API contracts (backend ⇄ frontend)

Source of truth for the frontend. Implemented in `backend/api/app.py` and `backend/api/game.py`,
exercised by `tests/test_api.py`. Real captured responses and WebSocket traces live in
`docs/api-fixtures/` (regenerate with `python -m scripts.make_fixtures`). The interactive schema is
at `http://127.0.0.1:8000/docs` while the server runs.

- Base URL: same origin. Use relative URLs (`/api/...`, `/ws/game`). `vite dev` proxies both to
  `127.0.0.1:8000` (`frontend/vite.config.ts`); in production FastAPI serves `frontend/dist` at `/`.
- JSON everywhere. Times are Unix seconds (float). No authentication: the server binds to localhost.
- Errors: HTTP 4xx with `{"detail": "human readable message"}` (422 validation errors may carry a
  list in `detail`). 404 unknown id, 409 state conflict (limit reached, wrong status), 422 bad input.
- Nothing is mocked server-side. If a field is `null`, the data does not exist yet; show an empty
  state, never a placeholder number.

## 1. Architecture you render against

The **server owns the simulation**. One Python engine (`backend/game/engine.py`) is used for human
play, AI play, competitions, training and evaluation, at a fixed 30 ticks per second. The browser is a
renderer plus an input device:

- The client opens `/ws/game`, sends `start`, and receives a `state` snapshot after every simulated
  tick (30 per second at 1x speed; at higher AI playback speeds several ticks may be folded into one
  message, at most 60 messages per second).
- Human input is a `flap` message. It is applied on the next tick (≤ 33 ms). Give instant visual
  feedback locally (wing animation, sound) when the key goes down; the authoritative position follows.
- Render at display refresh rate. Between snapshots you may interpolate or extrapolate for smoothness
  using `vy`, `gravity`, `speed` and `tick_rate` from `ready.config`. The latest snapshot always wins.
  Never run game rules (collision, scoring) in the browser.
- Rendering speed never affects outcomes: the same `seed` and inputs give the same match.

### World geometry (all values in world pixels, from `ready.config`)

- y grows **downward**: `0` is the ceiling, `world_height` (512) is the floor. Touching either ends
  the bird's run.
- The visible window is `view_width` (432) × `world_height`. Birds never move horizontally: every
  bird is drawn at `x = bird_x` (96) with a circular hitbox of radius `bird_radius` (12).
- `state.obstacles[]` are in **screen space**: `x` is the left edge, width is `obstacle_width` (60).
  Each obstacle is two solid blocks: from the ceiling down to `gap_top`, and from `gap_bottom` down
  to the floor. The opening is between them. `i` is a stable index (use it as the React/Pixi key).
- `state.scroll_x` is the total distance scrolled; use it for parallax backgrounds.
- Scale the 432×512 world uniformly to the available canvas; keep the aspect ratio.

## 2. WebSocket `/ws/game`

One socket hosts any number of consecutive matches. All messages are JSON objects with a `type`.
The handshake is refused (close code 1008) when the `Origin` header is neither the server's own origin
nor a local one (`localhost`, `127.0.0.1`), so other web pages cannot drive the server.

### Client → server

| type | fields | effect |
|---|---|---|
| `start` | `mode` (`"human"`, `"ai"`, `"human_vs_ai"`, `"ai_vs_ai"`), `models` (list of model ids: 0, 1, 1, 2 respectively), `seed` (int, optional: random if omitted), `difficulty` (`"easy"`/`"normal"`/`"hard"`, default `normal`), `limit_ticks` (optional competition limit, default 27000 = 15 min), `speed` (optional, AI-only modes), `view_width` (optional, 432–1296 world px, default 432: how much of the world each snapshot covers, so the renderer can draw a landscape scene; it changes `ready.config.view_width` and the obstacles sent, never the simulation) | Creates a match. Replies `ready` then one `state`. Can be sent at any time to replace the current match. |
| `flap` | – | Human bird flaps on the next tick. In phase `ready` it also starts the world. Ignored in modes without a human. |
| `pause` / `resume` | – | Freezes/unfreezes the simulation. Replies one `state` with the new `phase`. |
| `restart` | `same_seed` (bool, optional), `seed` (int, optional) | New match with the same mode/models/difficulty. New random seed unless `same_seed` or `seed` is given. This is also "rematch". |
| `set_speed` | `speed` (0.25–16) | Playback speed. Refused with an `error` while a human bird is alive; allowed in AI-only modes, replays, and in human modes **after the human has crashed** (fast-forward the rest of the AI's flight). |
| `replay` | `match_id`, `speed` (optional), `view_width` (optional) | Re-simulates a stored match from its recorded inputs. Behaves like an AI-only match (speed, pause). |
| `stop` | – | Ends the current match without saving. Replies `stopped`. |

### Server → client

**`ready`** (once per match)

```json
{"type":"ready","mode":"human_vs_ai","seed":5,"difficulty":"normal","in_distribution":true,
 "limit_ticks":27000,"replay_of":null,"personal_best":3,
 "config":{"tick_rate":30,"world_height":512.0,"view_width":432.0,"bird_x":96.0,"bird_radius":12.0,
           "start_y":256.0,"gravity":1200.0,"flap_velocity":360.0,"max_fall_speed":540.0,"speed":140.0,
           "obstacle_width":60.0,"obstacle_spacing":210.0,"first_obstacle_x":420.0,"gap_min":125.0,
           "gap_max":165.0,"gap_margin":50.0,"max_gap_shift":190.0},
 "players":[{"kind":"human","label":"You","model_id":null},
            {"kind":"ai","label":"ppo-r7","model_id":"ppo-r7","algo":"ppo","variant":null,
             "status":"evaluated","train_steps":150528,"diagnostic":"action_probs"}]}
```

- `players[i]` order is fixed for the match and matches `state.players[i]`. In `human_vs_ai` the
  human is index 0.
- `in_distribution` is `false` when the difficulty differs from the one models were trained on
  (`normal`). Show it; an agent on `hard` is being tested out of distribution.
- `personal_best`: best human score at this difficulty so far (`null` when no human plays).
- `diagnostic`: which diagnostic this model really exposes: `"q_values"` (DQN family) or
  `"action_probs"` (PPO).

**`state`** (every tick while running; once after `ready`, `pause`, `resume`, `set_speed`)

```json
{"type":"state","phase":"running","tick":123,"time":4.1,"speed":1.0,"scroll_x":574.0,
 "obstacles":[{"i":0,"x":-154.0,"gap_top":180.2,"gap_bottom":320.9},{"i":1,"x":56.0,"gap_top":210.0,"gap_bottom":350.5}],
 "players":[{"y":250.1,"vy":-120.0,"alive":true,"score":1,"distance":574.0,"flapped":false,"death":null},
            {"y":301.7,"vy":80.0,"alive":true,"score":1,"distance":574.0,"flapped":true,"death":null,
             "diag":{"obs":[0.59,0.15,0.1,0.24,0.5,0.1,0.17,0.27,0.59,0.55,0.29,0.7],"action":1,
                     "action_probs":[0.2,0.8]}}]}
```

- `phase`: `ready` (human modes: world frozen at tick 0 until the first `flap`), `running`,
  `paused`, `over`.
- `players[i].vy` in px/s (negative = rising). `flapped` is true on the tick that bird flapped.
  `death` is `null`, `"floor"`, `"ceiling"` or `"obstacle"`. `score` = obstacles fully cleared.
  `distance` = pixels travelled while alive (the ranking criterion).
- `diag` exists only for AI players that acted on this tick. `obs` is the 12-value observation
  (names in `GET /api/game/config` → `observation.names`), `action` is 0/1, plus **either**
  `q_values` `[no_flap, flap]` **or** `action_probs` `[no_flap, flap]`. Never display a diagnostic
  the message does not contain.
- A dead bird stays in `players` with `alive:false` and frozen `y`; the other bird keeps flying and
  the world keeps scrolling. The match ends when every bird is dead or `limit_ticks` is reached.

**`game_over`** (once, after the final `state` with `phase:"over"`)

```json
{"type":"game_over","result":{"match_id":42,"mode":"human_vs_ai","seed":5,"difficulty":"normal",
 "ticks":412,"limit_reached":false,"winner":1,"draw":false,"personal_best":3,"new_record":false,
 "players":[{"kind":"human","label":"You","model_id":null,"score":2,"distance":868.0,"ticks":186,
             "flaps":14,"death":"obstacle","died_tick":186},
            {"kind":"ai","label":"ppo-r7","model_id":"ppo-r7","algo":"ppo","variant":null,"status":"evaluated",
             "train_steps":150528,"diagnostic":"action_probs","score":7,"distance":1922.7,"ticks":412,
             "flaps":31,"death":"obstacle","died_tick":412}]}}
```

- `winner`: index of the player who travelled farther; `null` for single-player modes and draws
  (`draw:true` when two players tie exactly). `personal_best`/`new_record` only when a human played.
- The match is now stored (`GET /api/matches`, replayable). Replays answer with `"replay": true` and
  are not stored again.

**`error`** `{"type":"error","message":"..."}` — bad message, unknown/incompatible model, missing
checkpoint. The socket stays open. **`stopped`** `{"type":"stopped"}`.

## 3. REST: game, matches, models

| method & path | purpose |
|---|---|
| `GET /api/health` | `{ok, api_version, engine_version, env_version}` |
| `GET /api/game/config` | modes, the three difficulty configs, observation names, action meanings, speed range, limits |
| `GET /api/matches?mode=&limit=` | newest first: `{id, mode, seed, difficulty, participants, result, created_at}` (`result` = the `game_over.result` object) |
| `GET /api/matches/stats` | `{human:{games, best:{difficulty:score}, vs_ai:{wins,losses,draws}}, head_to_head:[{a,b,wins_a,wins_b,draws,n}], total}` |
| `GET /api/matches/{id}` | match row plus `replay` (`{seed, ticks, flap_ticks, ...}`); to *watch* it use the WS `replay` message |
| `POST /api/competitions/series` | body `{model_a, model_b, n_seeds (1–25, default 5), first_seed, difficulty, limit_ticks (default 4600 ≈ 100 obstacles)}` → runs headless on identical obstacle fields and returns `{n, wins_a, wins_b, draws, mean_score_a, mean_score_b, matches:[{match_id, seed, winner, draw, scores, ticks}]}`. Synchronous; can take several seconds. |
| `GET /api/models?status=&algo=&all=` | registry rows, newest first. Default: the **curated list** (the best checkpoint of each algorithm/variant, the promoted model, and models of runs still training). `all=true` returns every registered checkpoint. Row field `listed` is 1 or 0. |
| `GET /api/models/{id}` | row + `evals[]` + `checkpoint_exists` |
| `POST /api/models/{id}/status` | body `{status: "experimental"|"evaluated"|"archived"}`; promoted/training models are refused (409). Promotion is evidence-only and has no manual endpoint. |
| `POST /api/models/{id}/act` | body `{obs:[12 floats]}` → `{model_id, action, q_values|action_probs}` |
| `POST /api/models/{id}/evaluate` | body `{split:"validation"|"test", scenarios:[...], episodes, acknowledge_test_use}` → per-scenario stats. `test` needs `acknowledge_test_use:true` (409 otherwise; each use is counted). |

Model row:

```json
{"id":"xdqn-double-r31","run_id":31,"algo":"xdqn","variant":"double","experiment_id":4,"seed":0,
 "created_at":1.7e9,"updated_at":1.7e9,"code_version":"2ae67e4","train_steps":150000,
 "status":"experimental","checkpoint_path":"...","obs_dim":12,"obs_version":"1","n_actions":2,
 "env_version":"FlapLab-v1/engine1/obs1","config":{...},
 "metrics":{"final_val":{"n":50,"mean":181.2,"median":200,"max":200,"min":12,"std":44.1,
                         "success":{"10":1.0,"25":0.98,"50":0.94,"100":0.9}},
            "best_val_mean":200.0,"best_step":90000,"steps_to":{"10":40000,"25":50000,"50":50000,"100":60000},
            "auc":0.61,"wall_seconds":58.2,"steps":150000,"reason":"step budget reached"}}
```

`status`: `training` (run in progress, checkpoint already playable), `experimental` (unfinished
budget or unconfirmed variant), `evaluated`, `promoted` (the single best verified model), `archived`,
`failed`. Model ids are opaque strings; never parse them. A new checkpoint is `dqn-r<run>`, `ppo-r<run>`,
`xdqn-r<run>` (control) or `xdqn-<variant>-r<run>`; the best checkpoint of each algorithm/variant is renamed
to a curated name that starts with the algorithm: `PPO_Baseline`, `DQN_Baseline`, `XDQN_Control`,
`XDQN_PriorExplore`, … (`python -m scripts.lab curate`, also run automatically after each research session).
An unlisted model still resolves by id (`GET /api/models/{id}`, `start.models`).
Any model whose `checkpoint_exists` is true can be passed to `start`; `failed` ones may error.

## 4. REST: training

| method & path | purpose |
|---|---|
| `GET /api/algorithms` | `{algorithms:[{id, name, kind, default_hyperparameters, supports_variants}], variants:[{id, hypothesis, is_control}], variant_composition, thresholds, scenarios}` |
| `GET /api/profiles` | `{limits, profiles:{name:{description, steps, seeds, eval_every, eval_episodes, max_experiments, session_wall_seconds, ...}}, overridable:{key:{min,max}}}` |
| `POST /api/training/runs` | body `{algo:"dqn"|"ppo"|"xdqn", variant?, seed?, total_steps?, profile?, hyperparameters?, eval_every?, eval_episodes?, device?:"auto"|"cpu"|"cuda"}` → 201 run row. 409 when the concurrent-run or disk limit is reached. Training is a headless subprocess. |
| `GET /api/training/runs?status=&experiment_id=&limit=` | run rows, newest first |
| `GET /api/training/runs/{id}` | run row + `summary` + `metric_keys` |
| `GET /api/training/runs/{id}/metrics?keys=a,b&since_step=N` | `{key: [[step, value], ...]}`; poll about once per second with `since_step` = last step you hold |
| `GET /api/training/runs/{id}/evals` | validation evaluations (`purpose`: `periodic`, `final`, `manual`, `confirmatory`) |
| `GET /api/training/runs/{id}/log?tail=N` | `{run_id, lines:[...]}` raw trainer output |
| `POST /api/training/runs/{id}/stop` | safe stop: the trainer checkpoints and exits within ≤ 1000 steps → status `stopped`. 409 if not running. |
| `POST /api/training/runs/{id}/resume` | continues a `stopped`/`interrupted`/`limit` run from its checkpoint |

Run row (abridged): `{id, kind:"baseline"|"experimental", algo, variant, seed, profile, experiment_id,
arm, status, stop_requested, steps, total_steps, created_at, started_at, finished_at, wall_seconds,
code_version, env_version, device, hardware, model_id, error, group_key, config:{hyper, reward, game,
eval_every, eval_episodes, eval_max_score, limits, ...}}`.

Run `status`: `queued`, `starting`, `running`, `completed`, `stopped` (user), `limit` (a resource
limit ended it; `error` says which), `failed` (`error` holds the traceback), `interrupted` (process
vanished; resumable), `cancelled` (never launched because its session was stopped).

`summary`: `{curve:[[step, validation_mean], ...], final_mean, best_mean, auc, degradations,
steps_to:{"10":…,"25":…,"50":…,"100":…}, sps, wall_seconds, model_id}`.

Metric keys (present only if the algorithm logs them): always `sps` (env steps/s), `rss_mb`,
`gpu_mem_mb`, `cpu_percent`, `episodes`, `ep_score_mean`, `ep_len_mean`, `ep_return_mean` (rolling 100
training episodes), `val_mean`, `val_median`, `val_max` (at evaluation steps); DQN family: `loss`,
`epsilon`, and for xdqn `td_abs`, `q_mean` plus variant extras (`p_flap`, `rollbacks`, `lr`,
`aux_loss`, ...); PPO: `loss`, `value_loss`, `policy_gradient_loss`, `entropy_loss`, `approx_kl`,
`clip_fraction`, `explained_variance`. Read `metric_keys` instead of assuming.

Scores are obstacles cleared, capped at 200 per evaluation episode.

## 5. REST: comparison and leaderboard

A **group** = runs that differ only by training seed. A **benchmark** = `(env_version, total_steps)`.
Groups are only ever compared inside one benchmark; the API refuses to mix them.

| method & path | purpose |
|---|---|
| `GET /api/benchmarks` | `{benchmarks:[{env_version, total_steps, runs, groups}], metrics:{id: description}}` |
| `GET /api/compare?total_steps=&env_version=&groups=k1,k2` | `{benchmark, groups:[group]}`; defaults to the benchmark with most runs |
| `GET /api/compare/pair?a=<group_key>&b=<group_key>` | `{a, b, final_mean:{mean_a, mean_b, diff, ci:[lo,hi], rel_diff, prob_improvement, n_a, n_b}, auc:{...}, wall_ratio}`; 409 across benchmarks |
| `GET /api/leaderboard?metric=&total_steps=&env_version=` | `{metric, description, lower_is_better, benchmark, rows:[{rank?, label, group_key, algo, variant, kind, n_seeds, value, ci, final_mean, wall_seconds_mean, run_ids}]}`. Rows without data for the metric have `value:null` and no `rank`. |

Group object:

```json
{"group_key":"9f2c…","label":"xdqn+double","algo":"xdqn","variant":"double","kind":"experimental",
 "total_steps":150000,"env_version":"FlapLab-v1/engine1/obs1","n_seeds":5,"seeds":[0,1,2,3,4],"run_ids":[…],
 "final":{"mean":150.2,"ci":[101.0,193.5],"std":60.1,"values":[200,7.3,197.6,150,196]},
 "best":{…},"auc":{…},"seed_cv":0.4,"degradations_mean":1.6,
 "steps_to":{"10":{"reached":5,"of":5,"median":50000},"25":{…},"50":{…},"100":{…}},
 "wall_seconds_mean":55.1,"sps_mean":2722.0,
 "curve":[{"step":10000,"mean":0.2,"lo":0.0,"hi":0.5,"n":5}, …],
 "runs":[<run summary>, …],
 "generalization":{"default":{"mean":…,"ci":[…],"std":…,"values":[…]},"narrow_gap":{…}}}
```

- `curve` = mean validation score across seeds with a 95% bootstrap band (`lo`/`hi`): this is the
  sample-efficiency chart. `final`/`best`/`auc` carry per-seed `values` and a 95% CI of the mean.
- `generalization` holds held-out **test** results per scenario and is `{}` until a confirmatory
  evaluation exists. `default` is in-distribution; `narrow_gap`, `wide_spacing`, `tight_spacing`,
  `fast`, `slow`, `hard_combo` are out-of-distribution. Keep the two visually separate.
- Leaderboard metrics: `final_mean`, `best_mean`, `auc`, `steps_to_25`, `steps_to_50`,
  `steps_to_100` (lower is better), `test_mean`, `ood_mean`.

## 6. REST: research

| method & path | purpose |
|---|---|
| `GET /api/research/hypotheses` | all hypotheses with status |
| `GET /api/research/observations?profile=` | `{weaknesses:[{name, severity, evidence}], evidence:{…}, note}`: what the controller currently sees |
| `GET /api/research/queue?profile=` | ordered next experiments `[{hypothesis_id, title, variant, stage, reason}]` |
| `GET /api/research/experiments?session_id=&hypothesis_id=` | experiments, newest first, with `hypothesis_title` and `variant` |
| `GET /api/research/experiments/{id}` | experiment + `hypothesis` + `runs[]` + `report_markdown` |
| `GET /api/research/sessions` / `GET /api/research/sessions/{id}` | sessions (detail adds `experiments[]`) |
| `POST /api/research/sessions` | body `{profile, objective?, max_experiments?, steps?, seeds?, session_wall_seconds?, max_concurrent?, max_memory_mb?, max_gpu_mem_mb?, max_disk_mb?, early_stop_patience?, checkpoint_every?}` → 201 session. 409 if one is already active, 422 for out-of-range budgets (ranges in `GET /api/profiles` → `overridable`). |
| `POST /api/research/sessions/{id}/pause` | no new experiment is scheduled; the running one finishes |
| `POST /api/research/sessions/{id}/resume` | scheduling continues |
| `POST /api/research/sessions/{id}/stop` | nothing new is launched; running trainers checkpoint and exit; session → `stopped` |

Hypothesis: `{id, title, area, variant, targets, motivation, modification, expected_benefit,
failure_modes, criteria, baseline, prior_work, novelty:"known"|"unverified", budget, status, source,
created_at, updated_at}`. Status: `proposed` → `promising` | `requires_further_testing` |
`inconclusive` | `rejected` | `implementation_failed` | `errored`; `promising` → `confirmed` |
`not_confirmed` after the confirmation stage. "Accepted" modifications are `confirmed`; everything
else that was tested is shown as a result too, including failures.

Experiment: `{id, session_id, hypothesis_id, stage:"screening"|"intermediate"|"confirmation",
profile, status:"running"|"completed"|"failed"|"stopped"|"interrupted", decision, conclusion,
code_version, started_at, finished_at, error, report_path, config:{variant, steps, seeds, arms:{control:[run ids],
candidate:[run ids]}, reused, variant_source_sha, reason, observations}, result:{validation,
arms:{control|candidate:{completed, launched, final[], auc[], degradations[], wall[], ...}},
auc:{…compare…}, final_mean:{…compare…}, wall_ratio, held_out?, promotion?}}`.
Screening and intermediate stages are **exploratory**; only `confirmation` is evidence. Label them so.

Session: `{id, profile, objective, budgets:{…}, status:"queued"|"running"|"paused"|"completed"|
"stopped"|"failed"|"interrupted", control:"run"|"pause"|"stop", experiments_done, notes, started_at,
finished_at, error}`.

## 7. REST: system

`GET /api/status` (poll every 1–2 s for the Overview):

```json
{"time":1.7e9,
 "research":{"status":"idle|queued|running|paused","session":null,"active_experiment":null},
 "training":{"active_runs":[{"id":3,"algo":"ppo","variant":null,"seed":0,"status":"running","steps":41000,
                             "total_steps":150000,"experiment_id":null,"arm":"manual"}],"max_concurrent":4},
 "totals":{"experiments_completed":0,"experiments_failed":0,"runs_completed":15,"training_steps":2250000,
           "models":15,"matches":8,"held_out_test_uses":3},
 "best_verified_model":{"id":"ppo-r7","algo":"ppo","variant":null,"train_steps":150528,"status":"promoted",
                        "evidence":{"test_eval_id":61,"test_mean":198.2,"ood_mean":150.3,"replaced":null}},
 "best_score":200,
 "resources":{"cpu_percent":12.5,"cpu_count":28,"ram_used_mb":5100,"ram_total_mb":15900,
              "gpu":{"name":"…","utilization_percent":3,"memory_used_mb":3300,"memory_total_mb":16376},
              "disk_used_mb":120.5,"disk_limit_mb":5120,
              "device_benchmark":{"cpu":1137.9,"cuda":481.9,"chosen":"cpu","unit":"…"}},
 "limits":{"max_concurrent":4,"max_memory_mb":4096,"max_gpu_mem_mb":2048,"max_disk_mb":5120,
           "run_wall_seconds":1800,"early_stop_patience":0,"checkpoint_every":25000}}
```

`best_verified_model` and `gpu` may be `null`. `GET /api/events?since_id=&limit=` returns
`[{id, t, level:"info"|"warning"|"error", source, run_id, message}]`, newest first: the feed for
errors and warnings.
