# API fixtures

Real responses captured from this backend by `python -m scripts.make_fixtures` (trained models, real
runs, real research data at capture time). Use them for frontend development and tests.

- `ws_*.json`: WebSocket traces, a list of `{dir: "client"|"server", msg, note?}` in order.
  `ws_human*`: the "human" input is scripted (flagged with `note`). Long matches are thinned: after
  the first 300-400 states (and 60 states after the first crash) only every 10th state is kept, so
  ticks are not always consecutive. `ws_human_vs_ai` and `ws_ai_vs_ai` both contain the
  "one bird crashed, the other keeps flying" phase. `ws_ai_ppo` carries `action_probs`,
  `ws_ai_dqn` carries `q_values`.
- Everything else: one file per REST endpoint in `docs/api-contracts.md` (`error_*.json` are the
  bodies of 404/409/422 responses).
