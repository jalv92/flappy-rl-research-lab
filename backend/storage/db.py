"""SQLite persistence (stdlib sqlite3, WAL). Safe across the API process and trainer subprocesses.

Everything durable about the lab lives here; large artifacts (checkpoints, replays, reports)
are files referenced by path. FLAPLAB_HOME relocates experiments/ and checkpoints/ (tests use it).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_COLS = {"config", "hardware", "success", "scores", "metrics", "budgets", "budget", "result",
             "participants", "detail"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, kind TEXT, algo TEXT, variant TEXT, seed INTEGER, config TEXT, config_key TEXT,
  group_key TEXT, profile TEXT, experiment_id INTEGER, arm TEXT, status TEXT, stop_requested INTEGER DEFAULT 0, pid INTEGER,
  steps INTEGER DEFAULT 0, total_steps INTEGER, created_at REAL, started_at REAL, finished_at REAL,
  wall_seconds REAL DEFAULT 0, code_version TEXT, env_version TEXT, hardware TEXT, device TEXT,
  checkpoint_dir TEXT, model_id TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS metrics(run_id INTEGER, step INTEGER, key TEXT, value REAL);
CREATE INDEX IF NOT EXISTS metrics_idx ON metrics(run_id, key, step);
CREATE TABLE IF NOT EXISTS evals(
  id INTEGER PRIMARY KEY, run_id INTEGER, model_id TEXT, purpose TEXT, split TEXT, scenario TEXT, step INTEGER,
  n INTEGER, max_score INTEGER, mean REAL, median REAL, max REAL, min REAL, std REAL, success TEXT, scores TEXT,
  env_version TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS evals_idx ON evals(run_id, purpose, step);
CREATE TABLE IF NOT EXISTS models(
  id TEXT PRIMARY KEY, run_id INTEGER, algo TEXT, variant TEXT, experiment_id INTEGER, seed INTEGER,
  created_at REAL, updated_at REAL, code_version TEXT, config TEXT, train_steps INTEGER, metrics TEXT,
  checkpoint_path TEXT, obs_dim INTEGER, obs_version TEXT, n_actions INTEGER, env_version TEXT, status TEXT,
  listed INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS hypotheses(
  id TEXT PRIMARY KEY, title TEXT, area TEXT, motivation TEXT, modification TEXT, expected_benefit TEXT,
  failure_modes TEXT, criteria TEXT, baseline TEXT, budget TEXT, variant TEXT, prior_work TEXT, novelty TEXT,
  targets TEXT, status TEXT, source TEXT, created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY, session_id INTEGER, hypothesis_id TEXT, stage TEXT, profile TEXT, status TEXT,
  config TEXT, code_version TEXT, started_at REAL, finished_at REAL, result TEXT, decision TEXT,
  conclusion TEXT, error TEXT, report_path TEXT);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, profile TEXT, objective TEXT, budgets TEXT, status TEXT, control TEXT DEFAULT 'run',
  pid INTEGER, started_at REAL, finished_at REAL, experiments_done INTEGER DEFAULT 0, notes TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS matches(
  id INTEGER PRIMARY KEY, mode TEXT, seed INTEGER, difficulty TEXT, participants TEXT, result TEXT,
  replay_path TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, t REAL, level TEXT, source TEXT, run_id INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
"""


def home() -> Path:
    return Path(os.environ.get("FLAPLAB_HOME", ROOT))


def rel(path: str | Path) -> str:
    """Store paths relative to the lab home so a database can move between machines."""
    try:
        return Path(path).resolve().relative_to(home().resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else home() / p


def experiments_dir() -> Path:
    p = home() / "experiments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoints_dir() -> Path:
    p = home() / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def tx():
    con = sqlite3.connect(experiments_dir() / "lab.db", timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


SEED = ROOT / "seed"  # shipped snapshot: trained models + research records (see scripts/export_seed.py)


def seed_if_empty() -> bool:
    """First start of a fresh clone: copy the shipped lab (models, runs, experiments) into the runtime dirs."""
    target = experiments_dir() / "lab.db"
    if target.exists() or os.environ.get("FLAPLAB_SEED", "1") == "0" or not (SEED / "lab.db").exists():
        return False
    shutil.copytree(SEED / "checkpoints", checkpoints_dir(), dirs_exist_ok=True)
    if (SEED / "reports").exists():
        shutil.copytree(SEED / "reports", experiments_dir() / "reports", dirs_exist_ok=True)
    shutil.copyfile(SEED / "lab.db", target)  # last: its presence marks the seed as complete
    return True


def init() -> None:
    seed_if_empty()
    with tx() as con:
        con.executescript(SCHEMA)
        if "listed" not in {r["name"] for r in con.execute("PRAGMA table_info(models)")}:  # pre-curation databases
            con.execute("ALTER TABLE models ADD COLUMN listed INTEGER DEFAULT 1")


def _enc(v):
    return json.dumps(v) if isinstance(v, (dict, list, tuple)) else v


def _dec(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in JSON_COLS & d.keys():
        if isinstance(d[k], str):
            d[k] = json.loads(d[k])
    return d


def insert(table: str, **cols) -> int:
    keys = ", ".join(cols)
    marks = ", ".join("?" * len(cols))
    with tx() as con:
        return con.execute(f"INSERT OR REPLACE INTO {table}({keys}) VALUES({marks})",
                           [_enc(v) for v in cols.values()]).lastrowid


def update(table: str, id_, **cols) -> None:
    sets = ", ".join(f"{k}=?" for k in cols)
    with tx() as con:
        con.execute(f"UPDATE {table} SET {sets} WHERE id=?", [*(_enc(v) for v in cols.values()), id_])


def query(sql: str, params=()) -> list[dict]:
    with tx() as con:
        return [_dec(r) for r in con.execute(sql, params)]


def one(sql: str, params=()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def log_metrics(run_id: int, step: int, values: dict) -> None:
    rows = [(run_id, step, k, float(v)) for k, v in values.items() if v is not None and v == v]  # drop NaN
    with tx() as con:
        con.executemany("INSERT INTO metrics VALUES(?,?,?,?)", rows)


def event(level: str, source: str, message: str, run_id: int | None = None) -> None:
    insert("events", t=time.time(), level=level, source=source, run_id=run_id, message=message[:4000])


def kv_get(key: str, default=None):
    row = one("SELECT value FROM kv WHERE key=?", (key,))
    return json.loads(row["value"]) if row else default


def kv_set(key: str, value) -> None:
    with tx() as con:
        con.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (key, json.dumps(value)))


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """git short sha (+dirty). Recorded on every run, model and experiment."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip() or "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "backend", "configs"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def hardware() -> dict:
    import gymnasium
    import numpy
    import psutil
    import stable_baselines3
    import torch
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
        "packages": {"torch": torch.__version__, "numpy": numpy.__version__,
                     "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__},
    }
