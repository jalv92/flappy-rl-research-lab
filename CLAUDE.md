# Flappy RL Research Lab — project instructions

Arcade game (original, flap-through-gaps genre) + bounded autonomous RL research lab.
Read `docs/PROGRESS.md` first: what is verified, the research log and the known limits.

- **Ownership:** `frontend/` belongs to Codex and is changed only through `/codex-loop`
  (`docs/codex-integration.md`); after a frontend change run `npm run build` and commit `frontend/dist`. Claude owns `backend/`, `configs/`, `scripts/`, `tests/`, `docs/`.
- **One physics engine:** `backend/game/engine.py`. Never add a second implementation (not in the
  browser, not vectorised). Any trajectory-changing edit bumps `ENGINE_VERSION`.
- **Contract:** `docs/api-contracts.md` must match `backend/api/`; after changing an endpoint or a
  WebSocket message, update the doc, `tests/test_api.py`, and regenerate `docs/api-fixtures/`
  (`python -m scripts.make_fixtures`).
- **Science rules:** no fabricated metrics; failed experiments are recorded; decisions come from
  `backend/research/selection/decide.py`, not from enthusiasm; the test split is confirmatory only;
  no novelty claims without the literature check described in `docs/research-methodology.md`.
- **Budgets:** training is always bounded by `configs/profiles.toml`. Never start an unbounded run.
- Run from the project root with the venv: `.venv/bin/python -m pytest -q`,
  `.venv/bin/python -m scripts.lab <serve|bench|baselines|confirm-baselines|research|status>`.
- Temp files go in `tmp/` (gitignored). `experiments/` and `checkpoints/` are machine-local data;
  `seed/` is the shipped snapshot of them (`python -m scripts.export_seed` rebuilds it, sanitised).
- This repository is public: no absolute paths, machine names, personal data or internal notes in
  tracked files.
