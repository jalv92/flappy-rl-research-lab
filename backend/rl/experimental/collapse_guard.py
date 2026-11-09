"""Collapse guard: snapshot the networks whenever the smoothed training-episode length sets a new best; if it
later falls below half of that best, restore the snapshot and halve the learning rate.
Related ideas: checkpoint rollback, reduce-LR-on-plateau, conservative/trust-region updates. Novelty: UNVERIFIED.
extra: guard_drop (0.5), guard_min_len (150), guard_cooldown (5000 steps), guard_ema (0.1)."""
import copy

from backend.rl.agents.dqn import DQNAgent


class CollapseGuard(DQNAgent):
    name = "collapse_guard"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ema = self.best_ema = 0.0
        self.snapshot = None
        self.rollbacks = 0
        self.last_event = 0

    def on_episode(self, score, length):
        super().on_episode(score, length)
        x = self.cfg.extra
        a = float(x.get("guard_ema", 0.1))
        self.ema = (1 - a) * self.ema + a * length
        cooled = self.step - self.last_event >= int(x.get("guard_cooldown", 5000))
        if self.ema > self.best_ema and self.ema >= float(x.get("guard_min_len", 150)):
            self.best_ema = self.ema
            if cooled or self.snapshot is None:
                self.snapshot = (copy.deepcopy(self.net.state_dict()), copy.deepcopy(self.target.state_dict()))
                self.last_event = self.step
        elif self.snapshot and cooled and self.ema < float(x.get("guard_drop", 0.5)) * self.best_ema:
            self.net.load_state_dict(self.snapshot[0])
            self.target.load_state_dict(self.snapshot[1])
            for g in self.opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-5)
            self.ema = 0.75 * self.best_ema
            self.rollbacks += 1
            self.last_event = self.step

    def extra_metrics(self):
        return {**super().extra_metrics(), "rollbacks": self.rollbacks, "lr": self.opt.param_groups[0]["lr"],
                "len_ema": self.ema}
