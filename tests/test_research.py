import time

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.research.orchestration import controller
from backend.research.reporting import report
from backend.research.selection.decide import decide
from backend.rl.experimental import VARIANTS, source_sha, variant_class
from backend.storage import db


def cmp(a, b):
    return report.compare(a, b)


def test_decision_rule_needs_evidence():
    flat = cmp([10, 12, 11, 13], [11, 12, 10, 13])
    up = cmp([50, 55, 60, 58], [10, 12, 11, 13])
    down = cmp([1, 2, 1, 2], [10, 12, 11, 13])
    assert decide("screening", up, up)[0] == "promising"
    assert decide("screening", down, down)[0] == "rejected"
    assert decide("screening", flat, flat)[0] == "inconclusive"
    assert decide("screening", up, down)[0] == "inconclusive"  # mixed evidence is never promoted
    assert decide("confirmation", up, flat)[0] == "confirmed"
    assert decide("confirmation", flat, flat)[0] == "not_confirmed"
    trend = cmp([14, 9, 16, 12], [11, 12, 10, 9])  # better on average, CI crosses zero
    assert trend["ci"][0] < 0 < trend["diff"]
    assert decide("screening", trend, flat)[0] == "requires_further_testing"
    assert decide("intermediate", trend, flat)[0] == "inconclusive"  # no endless re-testing
    one_lucky_run = cmp([200, 190], [1, 2])
    assert decide("screening", one_lucky_run, one_lucky_run)[0] == "inconclusive"  # too few seeds


@pytest.mark.parametrize("variant", [*VARIANTS, "double+nstep+per", "dueling+softtarget+crash_focus"])
def test_every_variant_passes_preflight(variant):
    controller.preflight(variant)
    assert variant_class(variant).name == variant or variant == "control"
    assert len(source_sha(variant)) == 12


def test_budget_overrides_are_bounded():
    b = controller.budgets_for("quick_smoke", {"max_experiments": 2, "steps": None})
    assert b["max_experiments"] == 2 and b["steps"] == 4000
    with pytest.raises(ValueError):
        controller.budgets_for("quick_smoke", {"steps": 10**9})
    with pytest.raises(ValueError):
        controller.budgets_for("quick_smoke", {"python": 1})
    with pytest.raises(ValueError):
        controller.budgets_for("nope")


def test_bounded_research_cycle_records_everything(lab):
    """A real cycle: observe -> pick hypothesis -> validate -> train both arms -> evaluate -> decide -> document."""
    sid = controller.create_session("quick_smoke", "test", {"steps": 2000, "eval_every": 1000, "eval_episodes": 2})
    assert controller.run_session(sid) == "completed"
    s = db.one("SELECT * FROM sessions WHERE id=?", (sid,))
    assert s["experiments_done"] == 1 and "Experiment budget reached" in s["notes"]
    exp = db.one("SELECT * FROM experiments WHERE session_id=?", (sid,))
    assert exp["status"] == "completed" and exp["stage"] == "screening" and exp["decision"] == "inconclusive"
    assert len(exp["config"]["arms"]["control"]) == len(exp["config"]["arms"]["candidate"]) == 2
    assert exp["result"]["auc"]["n_a"] == 2 and exp["code_version"] and exp["config"]["variant_source_sha"]
    text = db.resolve(exp["report_path"]).read_text()
    assert "## Hypothesis" in text and "## Limitations" in text and "exploratory" in text
    # a wiring-check profile must not consume the hypothesis
    assert db.one("SELECT status FROM hypotheses WHERE id=?", (exp["hypothesis_id"],))["status"] == "proposed"
    # control runs are reused, not recomputed, by the next experiment with the same budget
    sid2 = controller.create_session("quick_smoke", "test", {"steps": 2000, "eval_every": 1000, "eval_episodes": 2})
    controller.run_session(sid2)
    exp2 = db.one("SELECT * FROM experiments WHERE session_id=?", (sid2,))
    assert set(exp2["config"]["arms"]["control"]) == set(exp["config"]["arms"]["control"])
    assert set(exp2["config"]["reused"]) >= set(exp["config"]["arms"]["control"])


def test_failed_implementation_is_recorded_as_a_result(lab, monkeypatch):
    monkeypatch.setitem(VARIANTS, "broken", "backend.rl.experimental.nope:Missing")
    db.insert("hypotheses", id="HX99", title="broken on purpose", variant="broken", status="proposed",
              targets="", created_at=time.time(), updated_at=time.time())
    b = controller.budgets_for("experimental_screening", {"steps": 2000, "seeds": 1})
    item = {"hypothesis_id": "HX99", "stage": "screening", "reason": "test"}
    exp = controller.run_experiment({"id": 0, "profile": "experimental_screening"}, item, b, lambda: False)
    assert exp["status"] == "failed" and exp["error"] and "pre-flight" in exp["conclusion"]
    assert db.one("SELECT status FROM hypotheses WHERE id='HX99'")["status"] == "implementation_failed"
    assert "HX99" not in [q["hypothesis_id"] for q in controller.queue(controller.observe(b), b)]  # never retried
    with db.tx() as con:
        con.execute("DELETE FROM hypotheses WHERE id='HX99'")


def test_queue_orders_followups_first_and_skips_closed_hypotheses(lab):
    b = controller.budgets_for("full_research")
    db.update("hypotheses", "H003", status="promising")
    db.update("hypotheses", "H006", status="requires_further_testing")
    db.update("hypotheses", "H009", status="rejected")
    try:
        q = controller.queue({"weaknesses": [{"name": "instability", "severity": 3.0}]}, b)
        assert [(i["hypothesis_id"], i["stage"]) for i in q[:2]] == [("H003", "confirmation"), ("H006", "intermediate")]
        assert "H009" not in [i["hypothesis_id"] for i in q]
        assert "instability" in q[2]["reason"]  # untested hypotheses that target an observed weakness come next
        only = controller.queue({"weaknesses": []}, controller.budgets_for("final_confirmation"))
        assert [(i["hypothesis_id"], i["stage"]) for i in only] == [("H003", "confirmation")]
    finally:
        for h in ("H003", "H006", "H009"):
            db.update("hypotheses", h, status="proposed")


def test_research_session_api_stop_launches_nothing_more(lab):
    with TestClient(app) as client:
        assert len(client.get("/api/research/queue").json()) >= 9
        assert client.post("/api/research/sessions", json={"profile": "nope"}).status_code == 422
        s = client.post("/api/research/sessions", json={"profile": "experimental_screening", "max_experiments": 3,
                                                        "steps": 200000, "seeds": 2})
        assert s.status_code == 201
        sid = s.json()["id"]
        assert client.post("/api/research/sessions", json={"profile": "quick_smoke"}).status_code == 409
        assert client.post(f"/api/research/sessions/{sid}/pause").json()["control"] == "pause"
        assert client.post(f"/api/research/sessions/{sid}/resume").json()["control"] == "run"
        t0 = time.time()  # let it get into training, then stop
        while time.time() - t0 < 60 and not client.get("/api/status").json()["training"]["active_runs"]:
            time.sleep(0.3)
        assert client.get("/api/status").json()["research"]["status"] in ("running", "queued")
        client.post(f"/api/research/sessions/{sid}/stop")
        while time.time() - t0 < 120 and client.get(f"/api/research/sessions/{sid}").json()["status"] != "stopped":
            time.sleep(0.3)
        detail = client.get(f"/api/research/sessions/{sid}").json()
        assert detail["status"] == "stopped" and detail["experiments_done"] <= 1
        assert not client.get("/api/status").json()["training"]["active_runs"]  # trainers checkpointed and exited
        runs = db.query("SELECT status, steps FROM runs WHERE experiment_id=?", (detail["experiments"][0]["id"],))
        assert runs and all(r["status"] in ("stopped", "cancelled") for r in runs)
        assert detail["experiments"][0]["status"] == "stopped" and detail["experiments"][0]["decision"] is None
        assert client.post(f"/api/research/sessions/{sid}/stop").status_code == 409


def test_brief_and_opt_in_agent_hook(lab, monkeypatch):
    """The coding-agent hand-off is a real mechanism: a brief file plus an opt-in command run once with a timeout."""
    from backend.research.hypotheses.library import seed_library
    seed_library()
    brief = controller.write_brief(controller.observe(controller.budgets_for("quick_smoke")))
    text = open(brief).read()
    assert "## Hypotheses so far" in text and "H001" in text and "Do not start training" in text
    monkeypatch.delenv("FLAPLAB_AGENT_CMD", raising=False)
    assert "no agent command configured" in controller.invoke_agent(brief)
    monkeypatch.setenv("FLAPLAB_AGENT_CMD", "head -n 1")  # stub agent: echoes the first line of the brief
    assert "exited 0: # Research brief" in controller.invoke_agent(brief)
    monkeypatch.setenv("FLAPLAB_AGENT_CMD", "definitely-not-a-command")
    assert "agent command failed" in controller.invoke_agent(brief)
