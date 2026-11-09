"""Hypothesis backlog authored by the research agent (Claude Code sessions).

Each entry is a falsifiable claim about ONE algorithmic change to the in-house DQN control, with the
closest prior work named. `novelty` is honest by construction: "known" = published technique re-tested
here; "unverified" = we have not established novelty and make no such claim. New hypotheses are appended
by later research sessions (see docs/research-methodology.md, "Invoking the coding agent").
`targets` are the weakness tags produced by the controller's OBSERVE step.
"""
from __future__ import annotations

import time

from backend.storage import db

BASELINE = "xdqn control: identical hyperparameters, step budget, seeds and evaluation schedule"
CRITERIA = ("Primary: validation AUC (sample efficiency) and final-policy validation mean, per training seed, "
            "candidate vs control. Promising only if a 95% bootstrap CI of the difference excludes 0 in the "
            "favourable direction and no primary metric is significantly worse. Secondary: degradation count, "
            "seed CV, wall-clock ratio.")
SOURCE = "claude-code-session-2026-11-09"

HYPOTHESES: list[dict] = [
    dict(id="H001", variant="double", area="loss_and_update_rules", targets="instability,final_performance",
         title="Double-DQN targets reduce value overestimation and the collapses it causes",
         motivation="The control's validation curves reach the score cap and then fall back to near zero. "
                    "Max-operator overestimation is a classic driver of such Q-value blow-ups.",
         modification="Select the bootstrap action with the online network and evaluate it with the target network.",
         expected_benefit="Fewer catastrophic degradations and a higher final-policy mean at equal steps.",
         failure_modes="With only two actions and sparse rewards the overestimation bias may be small; no effect.",
         prior_work="van Hasselt, Guez & Silver 2016 (Double DQN).", novelty="known"),
    dict(id="H002", variant="softtarget", area="loss_and_update_rules", targets="instability,seed_variance",
         title="Polyak-averaged targets smooth the moving target and stabilise late training",
         motivation="Hard target copies every 1000 steps change the regression target abruptly; collapses "
                    "often appear right after the policy first becomes good.",
         modification="Replace periodic hard target syncs with per-update Polyak averaging (tau=0.005).",
         expected_benefit="Fewer degradations and lower between-seed variance of the final policy.",
         failure_modes="Slower target tracking may delay early learning (lower AUC).",
         prior_work="Lillicrap et al. 2016 (DDPG soft updates).", novelty="known"),
    dict(id="H003", variant="nstep", area="hybrid_methods", targets="sample_efficiency",
         title="3-step returns propagate the sparse obstacle reward faster",
         motivation="Reward arrives once every ~45 ticks; one-step bootstrapping needs many sweeps to carry it back.",
         modification="Store 3-step returns (uncorrected) with gamma^n bootstrapping.",
         expected_benefit="Higher AUC / fewer steps to reach validation mean 25 and 50.",
         failure_modes="Off-policy bias from uncorrected multi-step targets under epsilon-greedy data; may "
                       "destabilise near the cap.",
         prior_work="Sutton 1988; Hessel et al. 2018 (Rainbow).", novelty="known"),
    dict(id="H004", variant="collapse_guard", area="adaptive_schedules", targets="instability,final_performance",
         title="Rolling back to the best snapshot and halving the learning rate when training collapses",
         motivation="Collapses are detectable online: the smoothed training-episode length drops sharply. "
                    "A reliable progress indicator can gate the optimiser instead of a fixed schedule.",
         modification="Snapshot networks at each new best smoothed episode length; when it falls below half of "
                      "that best, restore the snapshot and halve the learning rate (5000-step cooldown).",
         expected_benefit="Higher final-policy mean and fewer degradations at the same step budget.",
         failure_modes="The indicator is noisy under epsilon-greedy; spurious rollbacks waste steps; the lower "
                       "learning rate can freeze a mediocre policy.",
         prior_work="Related: checkpoint rollback, reduce-LR-on-plateau, trust-region ideas. No specific prior "
                    "paper identified for this exact rule.", novelty="unverified"),
    dict(id="H005", variant="prior_explore", area="exploration", targets="sample_efficiency,seed_variance",
         title="Drawing exploratory actions from a survivor action prior instead of uniformly",
         motivation="A uniform random action flaps 50% of the time, which ends the episode at the ceiling within "
                    "a few ticks; almost all early experience is the same uninformative crash.",
         modification="Epsilon-greedy's random actions use the flap frequency observed in the agent's own "
                      "longer-than-median episodes (starts at 0.5, i.e. identical to the control).",
         expected_benefit="Longer early episodes, earlier first obstacle clears, higher AUC.",
         failure_modes="The prior can lock onto a degenerate frequency; less state coverage near the ceiling.",
         prior_work="Related: action priors; temporally-extended exploration (ez-greedy, Dabney et al. 2021).",
         novelty="unverified"),
    dict(id="H006", variant="per", area="experience_replay", targets="sample_efficiency",
         title="Prioritising high-TD-error transitions speeds up learning of rare events",
         motivation="Obstacle clears and crashes are a small fraction of transitions; uniform replay "
                    "mostly revisits uneventful flight.",
         modification="Proportional prioritised replay (alpha=0.6, beta annealed 0.4->1) with importance weights.",
         expected_benefit="Higher AUC at equal steps.",
         failure_modes="Extra wall-clock per update; priorities chase noise; may amplify instability.",
         prior_work="Schaul et al. 2016 (PER).", novelty="known"),
    dict(id="H007", variant="crash_focus", area="experience_replay", targets="sample_efficiency,instability",
         title="Oversampling the last steps before a crash teaches obstacle avoidance sooner",
         motivation="The decisions that matter are those in the ~0.5 s before a collision; they are identifiable "
                    "without TD errors.",
         modification="Half of each batch is sampled from transitions within 15 steps before a termination.",
         expected_benefit="Higher AUC; cheaper than PER (no priority maintenance).",
         failure_modes="Uncorrected sampling bias over-weights failure states and may distort Q-values elsewhere.",
         prior_work="Related: PER (Schaul et al. 2016); episodic backward update (Lee et al. 2019).",
         novelty="unverified"),
    dict(id="H008", variant="aux_dynamics", area="predictive_learning", targets="sample_efficiency,generalization",
         title="An auxiliary forward-dynamics loss shapes features before reward arrives",
         motivation="Before the first obstacle is cleared the reward signal is only the crash penalty; the "
                    "dynamics are learnable from every transition.",
         modification="Add a head predicting the scaled observation change from (features, action); loss weight 0.5.",
         expected_benefit="Higher AUC and better out-of-distribution scores via physics-aware features.",
         failure_modes="The auxiliary gradient competes with the TD loss in a small network; extra compute.",
         prior_work="Jaderberg et al. 2017 (UNREAL); Schwarzer et al. 2021 (SPR).", novelty="known"),
    dict(id="H009", variant="dueling", area="policy_architectures", targets="sample_efficiency,final_performance",
         title="Dueling heads learn state value where the action rarely matters",
         motivation="In most states far from an obstacle both actions are near-equivalent in the short term.",
         modification="Separate value and advantage heads with mean-subtracted advantages.",
         expected_benefit="Higher AUC and final mean.",
         failure_modes="With two actions the decomposition adds little.",
         prior_work="Wang et al. 2016 (Dueling DQN).", novelty="known"),
]


def seed_library() -> int:
    """Insert library entries that are not yet in the database. Never overwrites recorded status/decisions."""
    have = {h["id"] for h in db.query("SELECT id FROM hypotheses")}
    now, n = time.time(), 0
    for h in HYPOTHESES:
        if h["id"] not in have:
            db.insert("hypotheses", **h, criteria=CRITERIA, baseline=BASELINE, budget={}, status="proposed",
                      source=SOURCE, created_at=now, updated_at=now)
            n += 1
    return n
