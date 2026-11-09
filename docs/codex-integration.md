# Codex integration: who owns what, and the frontend brief

## Ownership

| Area | Owner |
|---|---|
| `frontend/` (everything visual: design system, game renderer, all screens, assets, frontend tests) | **Codex** |
| `backend/`, `configs/`, `scripts/`, `tests/`, `docs/` | Claude Code |
| `docs/api-contracts.md`, `docs/api-fixtures/` | Claude Code (the shared interface; Codex reports gaps instead of working around them) |

Collaboration runs through [claude-codex-bridge](https://github.com/jalv92/claude-codex-bridge) (its
`/codex-loop` command). Claude publishes a task + contract, Codex implements in `frontend/`, Claude
runs the live backend/frontend integration tests and sends corrections back. Neither side edits the
other's directory; a missing or wrong API is reported in the reply so the backend can be fixed.

What Claude put in `frontend/` before the first hand-off (plumbing only, no UI): `package.json` with
an empty app and pre-installed dependencies (so no network approval is needed for a first build),
`package-lock.json`, and `vite.config.ts` with the `/api` and `/ws` dev proxy. Codex may change all of
it. Pre-installed: react, react-dom, react-router-dom, zustand, recharts, uplot, pixi.js, motion,
lucide-react, tailwindcss + @tailwindcss/vite, vite, typescript, vitest, jsdom, @testing-library/*.
These are options, not decisions: pick what fits, ignore the rest, add others if needed.

## Running the backend

```bash
.venv/bin/python -m scripts.lab serve          # http://127.0.0.1:8000, OpenAPI at /docs
cd frontend && npm run dev                     # http://localhost:5173, proxies /api and /ws
cd frontend && npm run build                   # FastAPI then serves frontend/dist at http://127.0.0.1:8000/
```

If the sandbox cannot reach the live server, develop against `docs/api-fixtures/`: real responses for
every REST endpoint and complete WebSocket traces for each game mode, captured from this backend with
real trained models. Claude runs the live integration pass after each round.

## Product brief

Two faces of one product, sharing one design language:

1. **The arcade game**: a colourful, polished flap-through-the-gaps game that feels like a finished
   indie title, not a debug view.
2. **The laboratory**: a professional dark-mode research dashboard with strong information
   hierarchy and restrained accent colours.

**Originality is a requirement.** This is an original game in the tap-to-flap genre. Design original
characters, obstacles, scenery, typography and a title of your own for the arcade game. Do not
reproduce the look, characters, obstacle art, fonts, sounds or name of any existing game. All assets
must be created for this project (SVG, Canvas or Pixi drawing code, or generated images you own).

### Game screens (all driven by the `/ws/game` protocol in `docs/api-contracts.md`)

- **Main menu** with navigation to every mode and to the lab. The game must remain fully usable
  with the lab screens closed.
- **Human**: Space, mouse click/tap and a configurable alternative key flap; pause; restart. Show
  score, personal record (`personal_best`, `new_record`), distance, a "get ready" state (phase
  `ready`), a game-over screen with restart, optional session statistics.
- **AI**: choose a model from `GET /api/models` (show algorithm, variant, training steps, status).
  Show score, best score this session, playback speed control (`set_speed`), pause/restart, and a
  diagnostics panel: current observation values with their names, the chosen action, and the
  Q-values **or** action probabilities exactly as present in `players[i].diag`. If a diagnostic is
  absent, do not draw it.
- **Human vs AI** (flagship): two visually distinct birds with clear labels in one world; live
  scores and relative progress that do not cover the play area; when one bird crashes the other
  keeps flying (grey out / mark the crashed one); a satisfying winner announcement with final
  statistics and a rematch button (`restart`). Match history from `GET /api/matches?mode=human_vs_ai`
  and `GET /api/matches/stats`.
- **AI vs AI**: pick two models, watch them in one world, show names, scores, progress, winner.
  Also a "series" action (`POST /api/competitions/series`) to compare them over several seeds with a
  per-seed table and totals.
- **Replay viewer**: list stored matches, replay any of them (`replay` message) with speed and pause.
- **Settings**: alternative flap key, reduced motion, sound on/off if you add sound. Persist locally.
- Show `in_distribution:false` clearly when an AI plays on a difficulty it was not trained on.

### Laboratory screens

- **Overview** (`GET /api/status`): research status, active experiment, completed experiments, best
  verified model, best score, total training steps, resource usage (CPU, RAM, GPU, disk vs limit).
- **Live training** (`/api/training/*`): start a run (algorithm, variant, seed, steps or profile),
  stop it safely, resume it; live charts for episode score/return, validation score, loss,
  throughput; active configuration; status; errors and warnings (`GET /api/events`, run `error`).
- **Algorithm comparison** (`GET /api/compare`, `/api/compare/pair`): baselines vs experimental
  groups inside one benchmark; sample-efficiency curves with the confidence band; final-score
  comparison with CIs; generalization (in-distribution vs out-of-distribution, visually separated);
  resource usage (wall-clock, steps/s).
- **Research history** (`/api/research/hypotheses`, `/experiments`, `/experiments/{id}`): every
  hypothesis with motivation, modification, expected benefit, failure modes, prior work, novelty
  status and current status; accepted **and** rejected/failed outcomes; the experiment report
  (`report_markdown`); code versions. Mark exploratory stages as exploratory.
- **Leaderboard** (`GET /api/benchmarks`, `/api/leaderboard`): selectable benchmark and metric; one
  benchmark at a time; show seeds and CI; rows without data are listed unranked.
- **Model viewer** (`GET /api/models`, `/api/models/{id}`): filter by status/algorithm, show
  metadata, metrics and evaluations, archive/restore, and "watch it play" (opens AI mode with it).
- **Research control centre** (`/api/research/sessions`, `/queue`, `/observations`, `/api/profiles`):
  start a bounded session (profile + budget overrides within the advertised ranges), pause, resume,
  stop; show the queue with the reason for each item and the current observations.

Every control must call the real endpoint and reflect the real result (including 409/422 messages).
No decorative buttons, no invented numbers, no client-side fake data outside tests. Every list needs
loading, empty and error states. Desktop first (≥ 1280 px), usable down to tablet width.

### Frontend tests (Codex)

`npm test` must cover at least: the WebSocket client/state reducer against the recorded traces in
`docs/api-fixtures/ws_*.json` for all four modes (including "one bird dead, the other still
flying"), diagnostics rendering for `q_values` vs `action_probs`, and the leaderboard/compare
adapters against the REST fixtures. `npm run build` must pass with TypeScript strict.

## Acceptance checks (bounded)

```bash
cd frontend && npm run build
cd frontend && npm test
```

Claude then verifies against the live backend: each game mode end to end in a browser, training
start/stop from the dashboard, model selection, research session start/pause/stop, persistence
after a restart.
