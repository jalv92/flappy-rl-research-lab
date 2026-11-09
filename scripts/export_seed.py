"""Build the shipped snapshot in seed/ from the current lab, so a fresh clone starts with trained models.

    python -m scripts.export_seed

What goes in: every completed run with its metrics and evaluations, the model registry, hypotheses,
experiments, research sessions and experiment reports, plus the best policy file of each model.
What stays out: matches and replays (each player starts their own history), events, resumable
training state and replay buffers, unfinished runs, process ids and machine-specific paths.
`backend.storage.db.seed_if_empty()` copies the snapshot into experiments/ and checkpoints/ on first start.
"""
from __future__ import annotations

import shutil
import sqlite3

from backend.storage import db

OUT = db.SEED


def main() -> None:
    db.init()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "checkpoints").mkdir(parents=True)
    (OUT / "reports").mkdir()
    src = sqlite3.connect(db.experiments_dir() / "lab.db")
    con = sqlite3.connect(OUT / "lab.db")
    src.backup(con)
    src.close()
    con.row_factory = sqlite3.Row
    home = str(db.home().resolve())

    # --- only finished science
    dropped = [r["id"] for r in con.execute("SELECT id FROM runs WHERE status<>'completed'")]
    if dropped:
        marks = ",".join("?" * len(dropped))
        for table, col in (("metrics", "run_id"), ("evals", "run_id"), ("models", "run_id"), ("runs", "id")):
            con.execute(f"DELETE FROM {table} WHERE {col} IN ({marks})", dropped)
    con.execute("DELETE FROM matches")
    con.execute("DELETE FROM events")
    con.execute("DELETE FROM kv WHERE key='device_benchmark'")  # measured per machine
    con.execute("UPDATE runs SET pid=NULL, stop_requested=0")
    con.execute("UPDATE sessions SET pid=NULL")

    # --- policies: copy the best file of every model, store portable paths
    copied = missing = 0
    for m in con.execute("SELECT id, run_id, checkpoint_path FROM models").fetchall():
        source = db.resolve(m["checkpoint_path"])
        relative = f"checkpoints/run_{m['run_id']:05d}/{source.name}"
        if source.exists():
            target = OUT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
        else:
            missing += 1
        con.execute("UPDATE models SET checkpoint_path=? WHERE id=?", (relative, m["id"]))
    for r in con.execute("SELECT id FROM runs").fetchall():
        con.execute("UPDATE runs SET checkpoint_dir=? WHERE id=?", (f"checkpoints/run_{r['id']:05d}", r["id"]))

    # --- reports
    for e in con.execute("SELECT id, report_path FROM experiments").fetchall():
        if e["report_path"] and db.resolve(e["report_path"]).exists():
            name = db.resolve(e["report_path"]).name
            shutil.copyfile(db.resolve(e["report_path"]), OUT / "reports" / name)
            con.execute("UPDATE experiments SET report_path=? WHERE id=?", (f"experiments/reports/{name}", e["id"]))

    # --- no machine paths anywhere in free text
    for table, cols in (("runs", ("error",)), ("sessions", ("notes", "error")),
                        ("experiments", ("conclusion", "error", "result", "config")), ("kv", ("value",))):
        for col in cols:
            con.execute(f"UPDATE {table} SET {col}=REPLACE(REPLACE({col}, ?, ''), ?, '') WHERE {col} LIKE ?",
                        (home + "/", home, f"%{home}%"))
    for report in (OUT / "reports").glob("*.md"):
        report.write_text(report.read_text().replace(home + "/", "").replace(home, ""))
    con.commit()
    leftovers = sum(con.execute(f"SELECT COUNT(*) FROM {t} WHERE {c} LIKE ?", (f"%{home}%",)).fetchone()[0]
                    for t, c in (("runs", "config"), ("runs", "hardware"), ("models", "config"), ("models", "metrics"),
                                 ("hypotheses", "motivation"), ("sessions", "budgets"), ("evals", "scores")))
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("VACUUM")
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("runs", "models", "metrics", "evals", "hypotheses", "experiments", "sessions", "matches")}
    listed = con.execute("SELECT COUNT(*) FROM models WHERE listed=1").fetchone()[0]
    con.close()
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**20
    print(f"seed/: {size:.1f} MB · policies copied {copied}, missing {missing} · listed models {listed}")
    print(f"rows: {counts} · dropped unfinished runs {dropped} · rows still containing a machine path: {leftovers}")


if __name__ == "__main__":
    main()
