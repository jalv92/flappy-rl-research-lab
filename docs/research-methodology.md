# Research methodology

The question the lab asks of every idea: **did this change actually improve reinforcement learning
on this task, under controlled conditions?** A failed experiment is a result and is stored like one.

## Roles

- **Research agent (Claude Code sessions).** Reads the evidence, writes hypotheses and their
  isolated implementations. This is intelligence applied in discrete, explicit sessions.
- **Research controller (`backend/research/orchestration/controller.py`).** A real, bounded process
  that selects, validates, trains, evaluates, decides and documents. It writes no algorithms.
- **Trainers.** One subprocess per run.

There is **no permanently running autonomous agent**. Autonomy means: within one session's budgets
the controller runs the whole experiment cycle without supervision.

### Invoking the coding agent

When the queue is empty the controller writes `experiments/research_brief.md` (observations, every
hypothesis and outcome so far, and the exact task for the next agent session). Two supported ways to
act on it:

1. **Manual session (default):** open Claude Code in this project and give it the brief.
2. **Opt-in automation:** set `FLAPLAB_AGENT_CMD` (for example
   `claude -p --permission-mode acceptEdits`) and add `invoke_agent = true` to the profile in
   `configs/profiles.toml`. The controller then runs that command once, with the brief on stdin and a timeout
   (`FLAPLAB_AGENT_TIMEOUT`, default 1800 s). The mechanism is covered by code and exercised with a
   stub command; it has **not** been run against a real agent in this repository, because that
   spends tokens and must be an explicit user decision.

Hypotheses and variants written by an agent land in git like any other code; the controller picks
them up through `VARIANTS` and the hypothesis library on its next session.

## Data splits

| Split | World seeds | Used for |
|---|---|---|
| training | ≥ 1,000,000 (random) | learning only |
| validation | 10,000–10,199 | learning curves, model selection, exploratory decisions |
| test (held out) | 20,000–20,199 | confirmatory evaluation only; every use is counted (`held_out_test_uses`) |
| OOD scenarios | test seeds, changed physics/layout | generalization, confirmation stage only |

Out-of-distribution scenarios keep the task recognisable: `narrow_gap`, `wide_spacing`,
`tight_spacing`, `fast`, `slow`, `hard_combo`. In-distribution (`default`) and OOD results are
always reported separately. Exploratory decisions never read the test split, so the controller
cannot overfit to it.

## Metrics (all on game score = obstacles cleared, capped at 200 per episode)

- **Sample efficiency:** validation mean every `eval_every` steps on 16 fixed worlds; `auc` =
  mean of that curve / 200; `steps_to` 10/25/50/100.
- **Final performance:** the final policy on 50 validation worlds (`final_mean`), with median,
  max, min, std and success rates at 10/25/50/100.
- **Stability:** `degradations` (evaluations that fall below half of an already-good running
  best), between-seed coefficient of variation, 95% bootstrap CIs over seeds.
- **Generalization:** held-out test mean per scenario (confirmation only).
- **Cost:** wall-clock per run and steps/s. A method that needs more compute is reported as such:
  a candidate above 1.5× the control's wall-clock is flagged "more sample-efficient, not more
  compute-efficient" even when it wins on steps.

Training reward is logged but never used to judge an agent.

## Controls

- Candidate and control share step budget, seeds, evaluation schedule, reward, environment version
  and hyperparameters; the only difference is the variant module (its source hash is recorded).
- Runs with an identical scientific identity (`config_key`) are reused instead of recomputed, so a
  control group is trained once per budget.
- Comparisons and rankings exist only inside one benchmark `(env_version, total_steps)`.
- A single high score proves nothing: decisions need ≥ 3 completed seeds per arm, and promotion
  needs a confirmation stage with fresh seeds.

## Novelty

Every hypothesis names its closest prior work and carries `novelty: known | unverified`. Variant
names (`collapse_guard`, `prior_explore`, …) are tracking labels, not claims. Nothing in this
repository is claimed to be a new algorithm. Before any such claim: a literature review, ablations
of every component, more seeds, more than one environment, and the limitations stated.

## Known limitations of the present setup

- One environment with a low-dimensional observation; results may not transfer.
- The score cap (200) saturates strong agents, so differences show mostly in sample efficiency
  and stability rather than peak score.
- Bootstrap CIs over 3–8 seeds are coarse; screening has a real false-positive and false-negative
  rate (see `experiment-protocol.md`).
- The `xdqn` control and SB3 `dqn` share hyperparameters but are different implementations
  (exploration sampling, warm-up); differences between them are implementation effects, not findings.
