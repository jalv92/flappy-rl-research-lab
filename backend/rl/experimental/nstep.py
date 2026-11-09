"""Multi-step returns (Sutton 1988; used in Rainbow, Hessel et al. 2018). extra: n_step (default 3)."""
from collections import deque

from backend.rl.agents.dqn import DQNAgent


class NStep(DQNAgent):
    name = "nstep"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n = int(self.cfg.extra.get("n_step", 3))
        self.pending = [deque() for _ in range(self.cfg.n_envs)]

    def store(self, obs, action, reward, next_obs, done, ended):
        g = self.cfg.gamma
        for i in range(len(obs)):
            q = self.pending[i]
            q.append((obs[i].copy(), action[i], float(reward[i])))
            # full window -> emit the oldest; episode boundary -> flush every pending (shorter) window
            while q and (len(q) == self.n or ended[i]):
                ret = sum(g ** k * t[2] for k, t in enumerate(q))
                self.add(q[0][0], q[0][1], ret, next_obs[i], done[i], g ** len(q), env=i)
                q.popleft()
