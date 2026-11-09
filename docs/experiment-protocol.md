# Experiment protocol

What one research session does, step by step. Code: `backend/research/orchestration/controller.py`.

```
python -m scripts.lab research --profile experimental_screening     # foreground
POST /api/research/sessions                                         # from the dashboard
```

## The cycle

1. **OBSERVE** (`observe`) Read the stored control-group results at the session's step budget and
   derive weaknesses with evidence: `instability` (catastrophic validation drops per run),
   `seed_variance` (CV of the final mean), `sample_efficiency` (AUC against the PPO baseline or 0.6),
   `final_performance` (final mean below 90% of the cap), `generalization` (OOD < 70% of
   in-distribution, once held-out data exists).
2. **HYPOTHESIZE / select** (`queue`) Order: confirmation of `promising` hypotheses (if the profile
   allows), then more seeds for `requires_further_testing`, then untested hypotheses ranked by how
   severely their `targets` are observed. `rejected`, `inconclusive`, `implementation_failed`,
   `errored`, `confirmed`, `not_confirmed` are never re-queued: retesting needs a new hypothesis.
3. **IMPLEMENT (check)** The variant must exist as an isolated module; its source hash and the git
   revision are recorded. The current best model is never touched by an experiment.
4. **VALIDATE** (`validate_variant`) In-process: construction, a 600-step rollout, action and
   Q-value shapes, finite losses/weights, checkpoint round-trip. Then a 3000-step run through the
   production trainer in a subprocess, and a load through the model registry. A failure here ends
   the experiment as `failed` and marks the hypothesis `implementation_failed`: recorded, not hidden.
5. **TRAIN** Control and candidate arms, one run per seed, launched under the global concurrency
   cap. Each run enforces its own wall-clock, memory and GPU limits and checkpoints periodically.
6. **EVALUATE** Per-seed `auc` and `final_mean` for both arms on the validation split, bootstrap
   95% CI of the difference (10,000 resamples), probability of improvement, degradation counts,
   wall-clock ratio.
7. **DECIDE** (`backend/research/selection/decide.py`), mechanically:

   | Condition | Decision |
   |---|---|
   | fewer than 3 completed seeds in an arm | `inconclusive` |
   | a primary metric's CI entirely > 0 and none entirely < 0 | `promising` (exploratory stages) / `confirmed` (confirmation) |
   | a primary metric's CI entirely < 0 and none entirely > 0 | `rejected` |
   | both directions significant | `inconclusive` (mixed) |
   | screening, no significance, but P(improvement) ≥ 0.75 with ≥ +10% on a metric and no opposite trend | `requires_further_testing` |
   | otherwise | `inconclusive` |
   | confirmation without a significant improvement | `not_confirmed` |

8. **DOCUMENT** The `experiments` row keeps config, arms, run ids, statistics, decision and
   conclusion; `experiments/reports/exp_NNNN.md` is the human-readable record, including the
   hypothesis text, per-seed values and limitations.
9. **ITERATE** Back to OBSERVE until a budget is exhausted, the queue is empty (a research brief is
   written) or the user stops the session.

## Stages (successive filtering)

| Stage | Seeds | Purpose | Reads test split |
|---|---|---|---|
| validation | 1 × 3000 steps | does it run at all | no |
| screening | profile `seeds` (training seeds 0…) | cheap exploratory filter | no |
| intermediate | same count again (seeds 50…), pooled with screening | resolve a non-significant trend | no |
| confirmation | ≥ 8 fresh seeds (100…), full budget | the only stage that counts as evidence | yes: `default` + 6 OOD scenarios, 50 episodes each, both arms |

Only hypotheses that earn it move to a more expensive stage; the full budget is never spent on
every idea. With 3 seeds, a percentile-bootstrap CI excludes zero roughly whenever all candidate
seeds beat all control seeds, so screening has an appreciable error rate in both directions. That is
acceptable for a filter and is exactly why `promising` is not `confirmed`.

**Promotion.** After a `confirmed` decision the candidate arm's champion (chosen on validation) is
compared with the current best verified model on held-out test mean, then OOD mean. It is promoted
only if strictly better; otherwise the incumbent stays. Baseline champions get the same treatment
via `python -m scripts.lab confirm-baselines`.

## Budgets (`configs/profiles.toml`)

| Control | Where enforced |
|---|---|
| `max_experiments`, `session_wall_seconds` | controller loop, before each experiment |
| `steps`, `seeds`, `eval_every`, `eval_episodes` | run creation |
| `max_concurrent` | `train.run_batch` and `POST /api/training/runs` (global count of live trainers) |
| `run_wall_seconds`, `max_memory_mb` | trainer, every 1000 steps → run status `limit` |
| `max_gpu_mem_mb` | `torch.cuda.set_per_process_memory_fraction` when a run uses CUDA |
| `max_disk_mb` | controller (session stops) and API (409 on new runs) |
| `early_stop_patience` | trainer; off by default so arms keep equal step budgets |
| `checkpoint_every` | trainer |

Profiles: `quick_smoke` (wiring check; never changes a hypothesis' status), `baseline_benchmark`,
`experimental_screening`, `full_research`, `final_confirmation`. API overrides are range-checked.

**Stop policy.** `stop` on a session: no further experiment or run is launched, queued runs become
`cancelled`, running trainers receive a stop request, checkpoint at their next log interval and exit
as `stopped`; the experiment is recorded as `stopped` with no decision. `pause`: the running
experiment finishes, nothing new is scheduled until `resume`. No paid or remote compute is used
anywhere.
