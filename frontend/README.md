# Aerie — Flappy RL Research Lab frontend

Original React + TypeScript + Vite arcade and research workspace. All game rules, model actions,
scores, match outcomes and research measurements come from the backend. The Canvas renderer only
interpolates received snapshots. Fixture data is imported exclusively by tests.

## Run

From `frontend/`:

```sh
npm run dev
npm run build
npm test
```

Vite proxies `/api` and `/ws` to the backend at `127.0.0.1:8000`. All routes use hash navigation,
so direct links work with the backend's static file server. The production output is `dist/`.
The integration backend already serves `frontend/dist` from disk, so rebuilding updates its UI
without restarting the server.

## Implemented screens

- Flight deck; Human; AI; Human vs AI; AI vs AI; replay archive/player; local settings.
- Lab overview; live training; algorithm comparison; research history and reports; leaderboard;
  model viewer; research control centre.
- Every screen is linked in the persistent navigation. The navigation becomes a drawer on tablets
  and smaller screens; settings are at the bottom of the sidebar.
- Human input: Space, pointer/tap, and a configurable letter/number. Launch focuses the flight area.
  Human vs AI retains a crashed bird while the opponent flies, with separate score cards, relative
  distance, final server statistics, a winner overlay and rematch. Records and match history are real.
- AI pickers default to non-archived models and prefer the promoted model. An explicit “watch it
  play” selection can include an archived model. Policy diagnostics show only the fields returned
  on the current tick. Out-of-distribution difficulty is marked after the server acknowledges it.
- AI-only and replay controls support speed, pause and restart. Human modes expose playback speed
  after every human has crashed while the match is still running or paused. AI vs AI includes bounded seeded
  series with totals, per-seed results and replay links.
- Training supports algorithm/variant/profile/seed/step/device selection, safe stop and resume,
  incremental measurements, validation, logs and events. No training starts automatically.
- Comparison preserves bootstrap bands and final CIs, with paired comparisons, resource cost,
  separate in-distribution/OOD evaluations, and exact chart data tables. Leaderboards preserve server
  ranking, null values, and absent CIs inside one selected benchmark.
- Research lists all outcomes, labels exploratory stages, renders reports safely, and exposes session
  budgets within server-advertised ranges, pause/resume/stop, queue reasons and observations.
- Models expose configuration, provenance, metrics, evaluations, checkpoint availability, archive /
  restore-as-experimental and watch actions. Promotion remains evidence-only.
- Settings persist the alternative flap key and reduced-motion preference. System reduced motion
  is respected. No audio was added, so there is no nonfunctional sound toggle.

## Art and source structure

- `public/aerie.svg`: original brand mark/favicon.
- `src/game/art.tsx`: original folded kite-birds and canyon illustration.
- `src/game/World.tsx`: original Canvas scenery, mineral gates and live bird drawing.
- `src/game/state.ts`, `useGame.ts`: authoritative snapshot reducer and WebSocket lifecycle.
- `src/api.ts`: abortable, non-overlapping REST polling and mutation/error handling.
- `src/lab/`: research screens and comparison/leaderboard adapters.
- `src/components/`: shared UI, SVG measurement charts and safe report rendering.
- `DESIGN.md`: visual direction, palette and data boundary.

No remote image or font dependencies. No game physics or mocked runtime data.

## Validation and handoff

`npm run build` passes with TypeScript `strict`, unused-local and unused-parameter checks enabled.
`npm test` passes: **118 tests in 6 files**, including all recorded game/replay traces, independent
survival after a crash, pause/resume/restart/error/disconnect state, DQN vs PPO diagnostics, benchmark
and leaderboard adapters, metric merging, all route renders, game inputs, replay speed, archived
model selection, bounded training requests, model archival, session controls, and real 404/409/422
error bodies. HTTP and WebSocket transports in UI tests are test doubles using the captured fixtures.

## Integration corrections (round 2)

Implemented inside `frontend/`:

- Playback speed stays hidden while a human is alive, appears after the human crashes, sends the
  documented `set_speed` message and preserves server error feedback.
- Winner, lead, series, rivalry and count copy use second-person and singular/plural agreement.
- All four modes and the replay viewer share a viewport-sized flight workspace. The world keeps
  its server-provided aspect ratio; live scores and transport controls stay outside it. The setup /
  diagnostics rail scrolls independently, with history and seeded series below the flight workspace.
- Eight comparison groups have distinct colours and dash patterns, with marker shapes and matching
  legend samples. Hovering a legend or selecting a focus group dims other lines and confidence bands;
  focus is also available on touch and keyboard. The exact-data table is retained.
- Safe-stop text depends on active run status and resets when a stopped run is polled, even if
  `stop_requested` remains set. Axis labels adapt their precision and deduplicate constant ranges.
- The navigation drawer covers tablet widths through 1000 px, traps focus, locks background scrolling,
  closes on route changes / Escape, and clears its scrim when resizing to desktop. Pointer tests cover
  mouse and touch input, primary versus secondary buttons, duplicate-click prevention, pause and death.

Validation: the strict production build and all 82 tests pass. New tests exercise post-crash speed
visibility / messages / server errors, winner and lead agreement, singular/plural series totals,
running → stop-requested → stopped polling, small/constant axis ranges, eight-series styling/focus,
pointer input and tablet drawer state. HTTP and WebSocket test doubles remain confined to tests.
The production bundle is rebuilt in `dist/`; no new image assets or dependencies were needed.

Claude subsequently verified round 2 in Chromium at 1440×900 and on an 820×1180 touch layout;
see the read-only project progress log. The viewport fit was later superseded by the larger minimum
world requested in round 3 below.


## Landscape flight area (round 3)

Implemented inside `frontend/`, retaining the existing visual design:

- Start and replay requests include `view_width = clamp(floor(world_height × stage_width /
  stage_height))`, using the server's `view_width_range`. Measurement excludes the stage border and
  temporary scrollbars. Older servers without a range retain their default width.
- Both restart buttons and result rematch/watch-again buttons measure the stage again. Live rematches
  send a fresh `start` with the acknowledged mode, models, difficulty and tick limit. The keep-seed
  checkbox still controls whether an explicit rematch reuses its seed; unsubmitted setup changes
  affect only “Start new flight.”
- Stage resizes settle for 250 ms; a change of at least 5% (and at least 24 world pixels) refreshes
  an untouched human preflight or restarts AI/replay playback at the beginning. Resize restarts retain
  seed/recording, playback speed and paused state. The contract has no in-place width update or seek.
- An active human match is never automatically restarted, including paused play, the interval before
  the first flap is acknowledged, and the surviving AI after the human crashes. Width changes wait
  for the next explicit flight. If a narrowed desktop cannot fit the old view at 480 px tall, it can
  scroll horizontally until then; scrollbar space is reserved to keep the full world height visible.
- The game heading and score cards are shorter, with all scores still outside the world. Stage height
  is independent of scores and transport. Above 1000 CSS px viewport width, the stage has at least
  480 px of content height and the world has a corresponding minimum width at its acknowledged aspect
  ratio. Short windows scroll the document. Tablet/phone layouts retain proportional fitting.
- The renderer continues uniform scaling and server snapshot interpolation. Mountain layers now
  extend across the entire negotiated window; the sun and sky already follow the world dimensions.
  No game rules, new image assets, dependencies, or changes outside `frontend/` were added.

Validation: strict `npm run build`; `npm test` has **118 passing tests**. Added coverage includes all
five viewers at modeled desktop content sizes, both range clamps, server-provided height/range,
restart/rematch/replay requests, resize debounce, pause/speed retention, human deferral, and the
minimum-height CSS contract plus uniform-scale calculations. The production `dist/` is rebuilt.

Pending browser verification: the permitted reproduction checks are build and Vitest only. jsdom
cannot measure real layout or paint Canvas. Claude still needs to measure >=80% world/stage width,
full world height, and >=480 px desktop world height at **1648×760** and **1440×900** in all four
modes and the replay viewer, and recheck tablet and physical pointer input. No live match, training,
research session or competition was started here. No blocking API gap was found; the older captured
config fixture lacks `view_width_range`, so frontend-only tests explicitly add that advertised field.
