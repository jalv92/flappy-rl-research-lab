"""Decision rule for experiments. Deliberately mechanical so that no result is promoted by enthusiasm.

Inputs are `report.compare()` dicts (candidate minus control, per training seed) for the two primary
metrics. A metric is *significantly* better/worse when its 95% bootstrap CI excludes zero.
Screening and intermediate stages are exploratory; only the confirmation stage (fresh seeds) yields
"confirmed"/"not_confirmed".
"""
from __future__ import annotations

MIN_SEEDS = 3  # below this a bootstrap over seeds is meaningless


def decide(stage: str, auc: dict, final: dict) -> tuple[str, str]:
    metrics = {"auc": auc, "final_mean": final}
    if min(auc["n_a"], auc["n_b"], final["n_a"], final["n_b"]) < MIN_SEEDS:
        return "inconclusive", f"fewer than {MIN_SEEDS} completed seeds in an arm; no statistical comparison possible"
    better = [k for k, c in metrics.items() if c["ci"][0] > 0]
    worse = [k for k, c in metrics.items() if c["ci"][1] < 0]
    desc = "; ".join(f"{k}: diff {c['diff']:+.3g} (95% CI {c['ci'][0]:+.3g}..{c['ci'][1]:+.3g}, "
                     f"P(improve)={c['prob_improvement']:.2f})" for k, c in metrics.items())
    if stage == "confirmation":
        if better and not worse:
            return "confirmed", f"significantly better on {', '.join(better)} with fresh seeds. {desc}"
        return "not_confirmed", f"the screening signal did not reproduce with fresh seeds. {desc}"
    if better and not worse:
        return "promising", f"significantly better on {', '.join(better)}, nothing significantly worse. {desc}"
    if worse and not better:
        return "rejected", f"significantly worse on {', '.join(worse)}, nothing significantly better. {desc}"
    if better and worse:
        return "inconclusive", f"mixed: better on {', '.join(better)}, worse on {', '.join(worse)}. {desc}"
    up = any(c["prob_improvement"] >= 0.75 and c["rel_diff"] >= 0.10 for c in metrics.values())
    down = any(c["prob_improvement"] <= 0.25 and c["rel_diff"] <= -0.10 for c in metrics.values())
    if stage == "screening" and up and not down:
        return "requires_further_testing", f"favourable trend but CIs include zero; more seeds needed. {desc}"
    return "inconclusive", f"no significant difference. {desc}"
