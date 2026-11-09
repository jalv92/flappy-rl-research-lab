"""Training runs: one bounded, resumable subprocess per run.

    python -m backend.rl.training.train RUN_ID [--resume]

A run row (created by `create_run`) fully specifies the experiment. The trainer logs metrics,
evaluates on the validation split, writes atomic checkpoints, honours stop requests and
resource limits, and registers the resulting model. Algorithms: "dqn"/"ppo" (Stable-Baselines3
baselines) and "xdqn" (in-house DQN + experimental variants).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import psutil
import torch

from backend.game.engine import GameConfig
from backend.models import registry
from backend.rl.agents.dqn import DQNConfig
from backend.rl.environments.env import ENV_VERSION, FlapEnv, RewardConfig
from backend.rl.evaluation.evaluate import THRESHOLDS, evaluate
from backend.rl.experimental import make_agent, parts, source_sha
from backend.storage import db

HYPER: dict[str, dict] = {
    # Stable-Baselines3 DQN (vanilla): same update-to-data ratio as xdqn (1 gradient step / 4 env steps)
    "dqn": dict(n_envs=8, net_arch=[128, 128], learning_rate=5e-4, buffer_size=100_000, learning_starts=2_000,
                batch_size=128, gamma=0.99, train_freq=1, gradient_steps=2, target_update_interval=1_000,
                exploration_fraction=0.2, exploration_final_eps=0.02),
    "ppo": dict(n_envs=8, net_arch=[64, 64], learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    "xdqn": {k: v for k, v in DQNConfig().to_dict().items()},
}
ALGOS = {
    "dqn": "DQN (Stable-Baselines3 baseline)",
    "ppo": "PPO (Stable-Baselines3 baseline)",
    "xdqn": "In-house DQN (control for experimental variants)",
}
DEFAULT_LIMITS = {"max_wall_seconds": 1800, "max_memory_mb": 4096, "max_gpu_mem_mb": 2048, "early_stop_patience": 0}
LOG_EVERY = 1000


def create_run(algo: str, variant: str | None = None, seed: int = 0, total_steps: int = 100_000,
               hyper: dict | None = None, reward: dict | None = None, profile: str | None = None,
               experiment_id: int | None = None, arm: str | None = None, eval_every: int = 10_000,
               eval_episodes: int = 16, checkpoint_every: int = 25_000, limits: dict | None = None,
               device: str = "auto") -> int:
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(ALGOS)}")
    if algo == "xdqn":
        variant = variant or "control"
        parts(variant)  # validates the name
    elif variant:
        raise ValueError("variants only apply to algo 'xdqn'")
    unknown = set(hyper or {}) - set(HYPER[algo])
    if unknown:
        raise ValueError(f"unknown hyperparameters for {algo}: {sorted(unknown)}")
    if not 1 <= total_steps <= 20_000_000:
        raise ValueError("total_steps must be within 1..20,000,000")
    science = {"hyper": {**HYPER[algo], **(hyper or {})}, "reward": RewardConfig(**(reward or {})).to_dict(),
               "game": GameConfig().to_dict(), "max_episode_steps": 5000, "eval_every": eval_every,
               "eval_episodes": eval_episodes, "eval_max_score": 200}
    source = source_sha(variant) if algo == "xdqn" else "sb3"
    ident = [algo, variant, total_steps, science, ENV_VERSION, source]
    group = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]  # same up to the seed
    # identical key == scientifically identical run (lets the orchestrator reuse control runs)
    key = hashlib.sha256(json.dumps([ident, seed], sort_keys=True).encode()).hexdigest()[:16]
    config = {**science, "checkpoint_every": checkpoint_every, "limits": {**DEFAULT_LIMITS, **(limits or {})},
              "device": device, "torch_threads": 1, "variant_source_sha": source}
    kind = "experimental" if algo == "xdqn" and variant != "control" else "baseline"
    return db.insert("runs", kind=kind, algo=algo, variant=variant, seed=seed, config=config, config_key=key, group_key=group,
                     profile=profile, experiment_id=experiment_id, arm=arm, status="queued", total_steps=total_steps,
                     created_at=time.time(), env_version=ENV_VERSION)


def launch(run_id: int, resume: bool = False) -> subprocess.Popen:
    logs = db.experiments_dir() / "logs"
    logs.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "backend.rl.training.train", str(run_id)] + (["--resume"] if resume else [])
    db.update("runs", run_id, status="starting", stop_requested=0)
    with open(logs / f"run_{run_id:05d}.log", "ab") as log:
        proc = subprocess.Popen(cmd, cwd=db.ROOT, stdout=log, stderr=subprocess.STDOUT)
    db.update("runs", run_id, pid=proc.pid)
    return proc


def request_stop(run_id: int) -> None:
    db.update("runs", run_id, stop_requested=1)


def active_runs() -> list[dict]:
    return db.query("SELECT * FROM runs WHERE status IN ('starting','running')")


def reconcile() -> int:
    """Mark runs whose process died without reporting (crash, reboot) as interrupted. They stay resumable."""
    n = 0
    for run in active_runs():
        alive = False
        try:
            alive = run["pid"] and "backend.rl.training.train" in " ".join(psutil.Process(run["pid"]).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        if not alive:
            db.update("runs", run["id"], status="interrupted", finished_at=time.time(),
                      error="trainer process disappeared; resume from the last checkpoint")
            n += 1
    return n


def run_batch(run_ids: list[int], max_concurrent: int = 4, should_stop=lambda: False, resume: bool = False) -> dict[int, str]:
    """Run queued runs to completion with a GLOBAL cap on concurrent trainer processes.
    When should_stop() turns true: nothing new is launched, running trainers checkpoint and exit."""
    pending, procs = list(run_ids), {}
    while pending or procs:
        stopping = should_stop()
        if stopping:
            for rid in pending:
                db.update("runs", rid, status="cancelled", finished_at=time.time(), error="session stopped before launch")
            pending = []
            for rid in procs:
                request_stop(rid)
        while pending and not stopping and len(active_runs()) < max_concurrent:
            rid = pending.pop(0)
            procs[rid] = launch(rid, resume)
        for rid, proc in list(procs.items()):
            if proc.poll() is not None:
                del procs[rid]
                row = db.one("SELECT status FROM runs WHERE id=?", (rid,))
                if row["status"] in ("starting", "running"):  # died without reporting
                    db.update("runs", rid, status="failed", finished_at=time.time(),
                              error=f"trainer exited with code {proc.returncode} before reporting")
        time.sleep(0.5)
    return {rid: db.one("SELECT status FROM runs WHERE id=?", (rid,))["status"] for rid in run_ids}


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"  # CPU fallback
    if requested != "auto":
        return requested
    bench = db.kv_get("device_benchmark")
    return bench["chosen"] if bench and (bench["chosen"] != "cuda" or torch.cuda.is_available()) else "cpu"


def benchmark_devices(seconds: float = 3.0) -> dict:
    """Measure real DQN update throughput per device instead of assuming CUDA wins. Result is cached in kv."""
    out = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        agent = make_agent("control", DQNConfig(buffer_size=10_000), 1, device)
        agent.size = 10_000
        agent.buf["obs"][:] = np.random.default_rng(0).random(agent.buf["obs"].shape, np.float32)
        agent.buf["next_obs"][:] = agent.buf["obs"]
        obs = agent.buf["obs"][:8]
        for _ in range(50):
            agent.update(0)  # warm-up
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < seconds:
            agent.update(0)
            agent.greedy(obs)
            n += 1
        if device == "cuda":
            torch.cuda.synchronize()
        out[device] = round(n / (time.perf_counter() - t0), 1)
    out["chosen"] = max((k for k in out), key=lambda k: out[k])
    out["unit"] = "DQN update+act iterations/s (batch 128, MLP 128x128, 1 torch thread)"
    db.kv_set("device_benchmark", out)
    return out


def atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class RunContext:
    """Everything a trainer needs besides the learning algorithm itself."""

    def __init__(self, run: dict, device: str, ckdir: Path, resume: bool):
        self.run, self.id, self.cfg, self.total = run, run["id"], run["config"], run["total_steps"]
        self.device, self.dir, self.resume = device, ckdir, resume
        state = json.loads((ckdir / "state.json").read_text()) if resume and (ckdir / "state.json").exists() else {}
        self.start_step = state.get("step", 0)
        self.best, self.best_step = state.get("best", -1.0), state.get("best_step", 0)
        self.prior_wall = state.get("wall_seconds", 0.0)
        self.solved_streak = 0
        self.t0 = time.time()
        self.step_now = self.start_step
        self.next_log = self.start_step + LOG_EVERY
        self.next_eval = (self.start_step // self.cfg["eval_every"] + 1) * self.cfg["eval_every"]
        self.next_ckpt = self.start_step + self.cfg["checkpoint_every"]
        self.last_log = (time.time(), self.start_step)
        self.scores, self.lengths, self.returns = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        self.episodes = 0
        self.proc = psutil.Process()
        self.status = "completed"
        self.reason: str | None = None
        self.model_registered = False

    def bind(self, act_fn, metrics_fn, save_last, save_best) -> None:
        self.act_fn, self.metrics_fn, self.save_last, self.save_best = act_fn, metrics_fn, save_last, save_best

    def episode(self, score: int, length: int, ret: float) -> None:
        self.scores.append(score)
        self.lengths.append(length)
        self.returns.append(ret)
        self.episodes += 1

    @property
    def wall(self) -> float:
        return self.prior_wall + time.time() - self.t0

    def _stop(self, status: str, reason: str) -> bool:
        self.status, self.reason = status, reason
        return False

    def eval_now(self, step: int, purpose: str = "periodic", n: int | None = None) -> dict:
        res = evaluate(self.act_fn, "default", "validation", n or self.cfg["eval_episodes"], self.cfg["eval_max_score"])
        db.insert("evals", run_id=self.id, model_id=registry.model_id(self.run), purpose=purpose, split="validation",
                  scenario="default", step=step, n=res["n"], max_score=res["max_score"], mean=res["mean"],
                  median=res["median"], max=res["max"], min=res["min"], std=res["std"], success=res["success"],
                  scores=res["scores"], env_version=ENV_VERSION, created_at=time.time())
        return res

    def checkpoint(self, step: int) -> None:
        self.save_last()
        (self.dir / "state.json").write_text(json.dumps(
            {"step": step, "best": self.best, "best_step": self.best_step, "wall_seconds": self.wall}))
        if not self.model_registered and not (self.dir / self.best_name).exists():
            self.save_best()  # make a loadable policy available early (dashboard can watch it play)
        if not self.model_registered:
            registry.register(self.run, self.dir / self.best_name, "training", step)
            self.model_registered = True

    def step(self, step: int) -> bool:
        """Call after every vectorised env step. Returns False when the trainer must stop."""
        self.step_now = step
        if step >= self.next_eval:
            self.next_eval += self.cfg["eval_every"]
            res = self.eval_now(step)
            db.log_metrics(self.id, step, {"val_mean": res["mean"], "val_median": res["median"], "val_max": res["max"]})
            if res["mean"] > self.best:
                self.best, self.best_step = res["mean"], step
                self.save_best()
            patience = self.cfg["limits"].get("early_stop_patience", 0)
            self.solved_streak = self.solved_streak + 1 if res["mean"] >= self.cfg["eval_max_score"] else 0
            if patience and self.solved_streak >= patience:
                return self._stop("completed", f"early stop: validation at cap for {patience} evaluations")
        if step >= self.next_ckpt:
            self.next_ckpt += self.cfg["checkpoint_every"]
            self.checkpoint(step)
        if step >= self.next_log:
            self.next_log += LOG_EVERY
            now = time.time()
            sps = (step - self.last_log[1]) / max(1e-9, now - self.last_log[0])
            self.last_log = (now, step)
            rss = self.proc.memory_info().rss / 2**20
            gpu = torch.cuda.memory_allocated() / 2**20 if self.device == "cuda" else 0.0
            m = {"sps": sps, "rss_mb": rss, "gpu_mem_mb": gpu, "cpu_percent": self.proc.cpu_percent(),
                 "episodes": self.episodes, **self.metrics_fn()}
            if self.scores:
                m.update(ep_score_mean=float(np.mean(self.scores)), ep_len_mean=float(np.mean(self.lengths)),
                         ep_return_mean=float(np.mean(self.returns)))
            bad = [k for k, v in m.items() if isinstance(v, float) and not np.isfinite(v)]
            if bad:
                raise FloatingPointError(f"non-finite training metrics {bad} at step {step}")
            db.log_metrics(self.id, step, m)
            row = db.one("SELECT stop_requested FROM runs WHERE id=?", (self.id,))
            db.update("runs", self.id, steps=step, wall_seconds=self.wall)
            lim = self.cfg["limits"]
            if row and row["stop_requested"]:
                return self._stop("stopped", "stop requested")
            if self.wall > lim["max_wall_seconds"]:
                return self._stop("limit", f"wall-clock limit {lim['max_wall_seconds']}s reached")
            if rss > lim["max_memory_mb"]:
                return self._stop("limit", f"memory limit {lim['max_memory_mb']} MB exceeded ({rss:.0f} MB)")
        return step < self.total or self._stop("completed", "step budget reached")

    def finish(self, error: str | None = None) -> None:
        step = self.step_now
        metrics = {}
        if error is None:
            self.checkpoint(step)
            if self.status == "completed":
                final = self.eval_now(step, "final", n=max(50, self.cfg["eval_episodes"]))
                if final["mean"] > self.best or not (self.dir / self.best_name).exists():
                    self.best, self.best_step = final["mean"], step
                    self.save_best()
                metrics["final_val"] = {k: v for k, v in final.items() if k != "scores"}
            curve = db.query("SELECT step, mean FROM evals WHERE run_id=? AND purpose='periodic' ORDER BY step", (self.id,))
            cap = self.cfg["eval_max_score"]
            metrics.update(
                best_val_mean=self.best, best_step=self.best_step,
                steps_to={str(t): next((c["step"] for c in curve if c["mean"] >= t), None) for t in THRESHOLDS},
                auc=float(np.mean([c["mean"] for c in curve]) / cap) if curve else 0.0,
                wall_seconds=self.wall, steps=step, reason=self.reason)
            kind_status = "experimental" if self.run["kind"] == "experimental" or self.status != "completed" else "evaluated"
            registry.register(self.run, self.dir / self.best_name, kind_status, step, metrics)
        else:
            self.status = "failed"
            db.event("error", "trainer", error[-3000:], self.id)
            if self.model_registered:
                db.update("models", registry.model_id(self.run), status="failed", updated_at=time.time())
        db.update("runs", self.id, status=self.status, steps=step, wall_seconds=self.wall, finished_at=time.time(),
                  error=error or (self.reason if self.status == "limit" else None))  # a user stop is not an error
        if self.status == "limit":
            db.event("warning", "trainer", self.reason, self.id)


def _envs(cfg: dict):
    game, reward = GameConfig(**cfg["game"]), RewardConfig(**cfg["reward"])
    return lambda: FlapEnv(game, reward, cfg["max_episode_steps"])


def train_xdqn(ctx: RunContext) -> None:
    cfg, run = ctx.cfg, ctx.run
    hyper = cfg["hyper"]
    dcfg = DQNConfig(**{**hyper, "hidden": tuple(hyper["hidden"])})
    agent = make_agent(run["variant"], dcfg, ctx.total, ctx.device, run["seed"])
    ctx.best_name = "best.pt"
    last, buf = ctx.dir / "last.pt", ctx.dir / "last.buffer.npz"
    step = ctx.start_step
    if ctx.resume and last.exists():
        agent.load_state(torch.load(last, map_location=ctx.device, weights_only=True)["agent"])
        if buf.exists():
            agent.load_buffer(np.load(buf))
    envs = [_envs(cfg)() for _ in range(dcfg.n_envs)]
    n = len(envs)
    obs = np.stack([e.reset(seed=run["seed"] * 100_003 + i + step)[0] for i, e in enumerate(envs)])
    ep_ret = np.zeros(n)
    last_m: dict = {}
    meta = {"variant": run["variant"], "hyper": hyper}

    def save_last():
        atomic_write(last, lambda f: torch.save({"agent": agent.state(), "step": step, **meta}, f))
        atomic_write(buf, lambda f: np.savez(f, **agent.buffer_arrays()))

    ctx.bind(agent.greedy, lambda: {**last_m, "epsilon": agent.epsilon(step), **agent.extra_metrics()}, save_last,
             lambda: atomic_write(ctx.dir / "best.pt", lambda f: torch.save({"net": agent.net.state_dict(), **meta}, f)))
    credit = 0.0
    while True:
        actions = agent.act(obs, step)
        next_obs, loop_obs = np.empty_like(obs), np.empty_like(obs)
        rew, done, ended = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, bool)
        for i, e in enumerate(envs):
            o2, r, term, trunc, info = e.step(int(actions[i]))
            next_obs[i], rew[i], done[i], ended[i] = o2, r, term, term or trunc
            ep_ret[i] += r
            if ended[i]:
                ctx.episode(info["score"], e.world.tick, ep_ret[i])
                agent.on_episode(info["score"], e.world.tick)
                ep_ret[i] = 0.0
                o2 = e.reset()[0]
            loop_obs[i] = o2
        agent.store(obs, actions, rew, next_obs, done, ended)
        obs = loop_obs
        step += n
        if step >= dcfg.learning_starts:
            credit += n / dcfg.train_freq
            while credit >= 1.0:
                last_m = agent.update(step)
                credit -= 1.0
        if not ctx.step(step):
            break


def train_sb3(ctx: RunContext) -> None:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    cfg, run = ctx.cfg, ctx.run
    algo = run["algo"]
    hyper = dict(cfg["hyper"])
    n_envs, net_arch = hyper.pop("n_envs"), hyper.pop("net_arch")
    env = make_vec_env(_envs(cfg), n_envs=n_envs, seed=run["seed"])
    cls = {"dqn": DQN, "ppo": PPO}[algo]
    ctx.best_name = "best.zip"
    last, buf = ctx.dir / "last.zip", ctx.dir / "last.buffer.pkl"
    resumed = ctx.resume and last.exists()
    if resumed:
        model = cls.load(last, env=env, device=ctx.device)
        if algo == "dqn" and buf.exists():
            model.load_replay_buffer(buf)
    else:
        model = cls("MlpPolicy", env, seed=run["seed"], device=ctx.device, verbose=0,
                    policy_kwargs={"net_arch": net_arch}, **hyper)

    @torch.no_grad()
    def act(obs):
        return model.policy._predict(torch.as_tensor(obs, device=model.device), deterministic=True).cpu().numpy()

    def metrics():
        m = {k.split("/", 1)[1]: float(v) for k, v in model.logger.name_to_value.items() if k.startswith("train/")}
        if algo == "dqn":
            m["epsilon"] = float(model.exploration_rate)
        return m

    def save_last():
        atomic_write(last, model.save)
        if algo == "dqn":
            atomic_write(buf, model.save_replay_buffer)

    ctx.bind(act, metrics, save_last, lambda: atomic_write(ctx.dir / "best.zip", model.save))

    class Hook(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "reward_components" in info:
                    ctx.episode(info["score"], info["episode"]["l"], info["episode"]["r"])
            return ctx.step(self.num_timesteps)

    remaining = ctx.total - (model.num_timesteps if resumed else 0)
    if remaining > 0:
        model.learn(remaining, callback=Hook(), reset_num_timesteps=not resumed, log_interval=10**9)
    ctx.step_now = model.num_timesteps


TRAINERS = {"dqn": train_sb3, "ppo": train_sb3, "xdqn": train_xdqn}


def main(run_id: int, resume: bool = False) -> str:
    db.init()
    run = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
    if run is None:
        raise SystemExit(f"run {run_id} does not exist")
    cfg = run["config"]
    torch.set_num_threads(cfg.get("torch_threads", 1))
    device = resolve_device(cfg["device"])
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["limits"]["max_gpu_mem_mb"] / total))
    ckdir = db.checkpoints_dir() / f"run_{run_id:05d}"
    ckdir.mkdir(parents=True, exist_ok=True)
    db.update("runs", run_id, status="running", pid=os.getpid(), started_at=run["started_at"] or time.time(),
              finished_at=None, code_version=db.code_version(), hardware=db.hardware(), device=device,
              checkpoint_dir=db.rel(ckdir), error=None)
    ctx = RunContext(run, device, ckdir, resume)
    error = None
    try:
        TRAINERS[run["algo"]](ctx)
    except KeyboardInterrupt:
        ctx.status, ctx.reason = "stopped", "interrupted by signal"
    except Exception:  # recorded, never hidden: a failed experiment is a result
        error = traceback.format_exc()
    try:
        ctx.finish(error)
    except Exception:
        db.update("runs", run_id, status="failed", finished_at=time.time(), error=traceback.format_exc()[-3000:])
        raise
    return ctx.status


if __name__ == "__main__":
    raise SystemExit(0 if main(int(sys.argv[1]), "--resume" in sys.argv) != "failed" else 1)
