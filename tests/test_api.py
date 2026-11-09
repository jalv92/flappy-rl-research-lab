"""Backend end-to-end tests through the real HTTP/WebSocket API (one per game mode + training + persistence)."""
import time

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.rl.environments.env import OBS_DIM


@pytest.fixture(scope="module")
def client(lab):
    with TestClient(app) as c:
        yield c


def play(ws, on_state=None, max_msgs=20000):
    """Drain a match until game_over. Returns (states, result)."""
    states = []
    for _ in range(max_msgs):
        msg = ws.receive_json()
        if msg["type"] == "state":
            states.append(msg)
            if on_state:
                on_state(msg)
        elif msg["type"] == "game_over":
            return states, msg["result"]
        elif msg["type"] == "error":
            raise AssertionError(msg["message"])
    raise AssertionError("match did not finish")


def test_health_and_game_config(client):
    assert client.get("/api/health").json()["ok"]
    cfg = client.get("/api/game/config").json()
    assert set(cfg["modes"]) == {"human", "ai", "human_vs_ai", "ai_vs_ai"}
    assert cfg["observation"]["dim"] == OBS_DIM == len(cfg["observation"]["names"])
    assert cfg["difficulties"]["normal"]["tick_rate"] == 30


def test_mode_human(client):
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start", "mode": "human", "seed": 7})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["seed"] == 7 and ready["players"][0]["kind"] == "human"
        first = ws.receive_json()
        assert first["phase"] == "ready" and first["tick"] == 0  # the world waits for the first flap
        ws.send_json({"type": "flap"})
        states, result = play(ws)
        assert states[0]["players"][0]["flapped"] and states[0]["players"][0]["vy"] < 0
        assert [s["tick"] for s in states[:-1]] == sorted({s["tick"] for s in states[:-1]})
        assert result["players"][0]["death"] == "floor" and result["winner"] is None and result["match_id"]
        assert result["personal_best"] >= result["players"][0]["score"] and "new_record" in result
        ws.send_json({"type": "restart", "same_seed": True})
        assert ws.receive_json()["seed"] == 7
    assert client.get("/api/matches", params={"mode": "human"}).json()[0]["id"] == result["match_id"]


@pytest.mark.parametrize("algo", ["xdqn", "dqn", "ppo"])
def test_mode_ai_uses_the_real_policy_and_exposes_only_real_diagnostics(client, lab, algo):
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start", "mode": "ai", "models": [lab[algo]], "seed": 11, "speed": 16,
                      "limit_ticks": 400})
        ready = ws.receive_json()
        assert ready["players"][0]["model_id"] == lab[algo] and ready["players"][0]["algo"] == algo
        states, result = play(ws)
    diag = next(s["players"][0]["diag"] for s in states if "diag" in s["players"][0])
    key = "action_probs" if algo == "ppo" else "q_values"
    assert set(diag) == {"obs", "action", key} and len(diag["obs"]) == OBS_DIM and len(diag[key]) == 2
    # the streamed action is exactly what the registered model returns for the streamed observation
    act = client.post(f"/api/models/{lab[algo]}/act", json={"obs": diag["obs"]}).json()
    assert act["action"] == diag["action"] and act[key] == pytest.approx(diag[key], abs=1e-5)
    assert result["players"][0]["ticks"] <= 400


def test_mode_human_vs_ai_shares_one_world_and_continues_after_a_crash(client, lab):
    def steer(msg):  # the "human": flap when low and falling
        me = msg["players"][0]
        if me["alive"] and me["y"] > 280 and me["vy"] >= 0:
            ws.send_json({"type": "flap"})

    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start", "mode": "human_vs_ai", "models": [lab["ppo"]], "seed": 5, "limit_ticks": 240})
        ready = ws.receive_json()
        assert [p["kind"] for p in ready["players"]] == ["human", "ai"]
        assert ws.receive_json()["phase"] == "ready"
        ws.send_json({"type": "flap"})
        states, result = play(ws, steer)
    assert all(len(s["players"]) == 2 and "obstacles" in s for s in states)  # one obstacle list for both birds
    ticks = [p["ticks"] for p in result["players"]]
    if ticks[0] != ticks[1]:
        loser = ticks.index(min(ticks))
        after = [s for s in states if not s["players"][loser]["alive"] and s["players"][1 - loser]["alive"]]
        assert len(after) >= 1 and after[-1]["tick"] > min(ticks)  # the world kept running for the survivor
        assert result["winner"] == 1 - loser and not result["draw"]
    else:
        assert result["draw"]
    assert result["match_id"] and result["seed"] == 5


def test_mode_ai_vs_ai_and_exact_replay(client, lab):
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start", "mode": "ai_vs_ai", "models": [lab["dqn"], lab["ppo"]], "seed": 9,
                      "speed": 16, "limit_ticks": 300})
        ready = ws.receive_json()
        assert [p["model_id"] for p in ready["players"]] == [lab["dqn"], lab["ppo"]]
        _, result = play(ws)
        ws.send_json({"type": "replay", "match_id": result["match_id"], "speed": 16})
        assert ws.receive_json()["replay_of"] == result["match_id"]
        _, again = play(ws)
    pick = lambda r: [(p["score"], p["ticks"], p["death"]) for p in r["players"]]
    assert pick(again) == pick(result) and again["replay"]
    detail = client.get(f"/api/matches/{result['match_id']}").json()
    assert detail["replay"]["seed"] == 9 and len(detail["replay"]["flap_ticks"]) == 2
    series = client.post("/api/competitions/series", json={"model_a": lab["dqn"], "model_b": lab["ppo"],
                                                           "n_seeds": 3, "limit_ticks": 200}).json()
    assert series["n"] == 3 and series["wins_a"] + series["wins_b"] + series["draws"] == 3
    again = client.post("/api/competitions/series", json={"model_a": lab["dqn"], "model_b": lab["ppo"],
                                                          "n_seeds": 3, "limit_ticks": 200}).json()
    assert [m["scores"] for m in again["matches"]] == [m["scores"] for m in series["matches"]]  # reproducible
    stats = client.get("/api/matches/stats").json()
    assert stats["total"] >= 8 and any(h["n"] >= 6 for h in stats["head_to_head"])


def test_game_errors_are_reported_not_faked(client, lab):
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "flap"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "start", "mode": "ai", "models": ["does-not-exist"]})
        assert "unknown model" in ws.receive_json()["message"]
        ws.send_json({"type": "start", "mode": "ai_vs_ai", "models": [lab["ppo"]]})
        assert "exactly 2" in ws.receive_json()["message"]
        ws.send_json({"type": "start", "mode": "human"})
        ws.receive_json(), ws.receive_json()
        ws.send_json({"type": "set_speed", "speed": 4})
        assert "fixed at 1x" in ws.receive_json()["message"]
    assert client.post("/api/models/nope/act", json={"obs": [0.0] * OBS_DIM}).status_code == 404
    assert client.post(f"/api/models/{lab['ppo']}/act", json={"obs": [0.0]}).status_code == 422


def test_models_evaluation_and_test_split_guard(client, lab):
    rows = client.get("/api/models").json()
    assert {lab["xdqn"], lab["dqn"], lab["ppo"]} <= {m["id"] for m in rows}
    mid = lab["ppo"]
    assert client.post(f"/api/models/{mid}/evaluate", json={"split": "test"}).status_code == 409
    res = client.post(f"/api/models/{mid}/evaluate", json={"scenarios": ["default", "fast"], "episodes": 3}).json()
    assert [r["scenario"] for r in res] == ["default", "fast"] and "scores" not in res[0]
    before = client.get("/api/status").json()["totals"]["held_out_test_uses"]
    ok = client.post(f"/api/models/{mid}/evaluate", json={"split": "test", "episodes": 2, "acknowledge_test_use": True})
    assert ok.status_code == 200
    assert client.get("/api/status").json()["totals"]["held_out_test_uses"] == before + 1
    assert len(client.get(f"/api/models/{mid}").json()["evals"]) >= 5
    assert client.post(f"/api/models/{mid}/status", json={"status": "archived"}).json()["status"] == "archived"
    assert client.post(f"/api/models/{mid}/status", json={"status": "evaluated"}).json()["status"] == "evaluated"
    assert client.post(f"/api/models/{mid}/status", json={"status": "promoted"}).status_code == 422


def test_training_can_be_launched_monitored_and_stopped(client, lab):
    assert client.post("/api/training/runs", json={"algo": "xdqn", "variant": "nope"}).status_code == 422
    run = client.post("/api/training/runs", json={"algo": "xdqn", "variant": "double", "seed": 21,
                                                  "total_steps": 200000, "eval_every": 2000, "eval_episodes": 2,
                                                  "hyperparameters": {"learning_starts": 500}})
    assert run.status_code == 201
    rid = run.json()["id"]
    t0 = time.time()
    while time.time() - t0 < 60:
        m = client.get(f"/api/training/runs/{rid}/metrics", params={"keys": "loss,sps"}).json()
        if len(m.get("loss", [])) >= 2:
            break
        time.sleep(0.3)
    assert len(m["loss"]) >= 2 and set(m) == {"loss", "sps"}
    since = m["loss"][-1][0]
    assert all(step > since for step, _ in client.get(f"/api/training/runs/{rid}/metrics",
                                                      params={"keys": "loss", "since_step": since}).json().get("loss", []))
    assert any(r["id"] == rid for r in client.get("/api/status").json()["training"]["active_runs"])
    assert client.post(f"/api/training/runs/{rid}/stop").json()["stop_requested"]
    while time.time() - t0 < 90 and client.get(f"/api/training/runs/{rid}").json()["status"] == "running":
        time.sleep(0.3)
    detail = client.get(f"/api/training/runs/{rid}").json()
    assert detail["status"] == "stopped" and 0 < detail["steps"] < 200000 and detail["summary"]["curve"]
    assert client.post(f"/api/training/runs/{rid}/stop").status_code == 409
    assert client.get(f"/api/models/{detail['model_id']}").json()["checkpoint_exists"]  # stopped run is watchable
    assert client.get(f"/api/training/runs/{rid}/log").status_code == 200
    assert {a["id"] for a in client.get("/api/algorithms").json()["algorithms"]} == {"dqn", "ppo", "xdqn"}
    assert "quick_smoke" in client.get("/api/profiles").json()["profiles"]


def test_comparison_and_leaderboard_never_mix_benchmarks(client, lab):
    bms = client.get("/api/benchmarks").json()["benchmarks"]
    assert any(b["total_steps"] == 3000 for b in bms)
    cmp = client.get("/api/compare", params={"total_steps": 3000}).json()
    assert {g["label"] for g in cmp["groups"]} >= {"dqn", "ppo", "xdqn"}
    assert all(g["total_steps"] == 3000 and g["curve"] and g["final"]["ci"] for g in cmp["groups"])
    lb = client.get("/api/leaderboard", params={"metric": "final_mean", "total_steps": 3000}).json()
    assert [r["rank"] for r in lb["rows"] if "rank" in r] == list(range(1, len(lb["rows"]) + 1))
    assert client.get("/api/leaderboard", params={"metric": "bogus", "total_steps": 3000}).status_code == 422
    a, b = cmp["groups"][0]["group_key"], cmp["groups"][1]["group_key"]
    pair = client.get("/api/compare/pair", params={"a": a, "b": b}).json()
    assert len(pair["final_mean"]["ci"]) == 2


def test_everything_survives_an_api_restart(client, lab):
    before = (len(client.get("/api/models").json()), client.get("/api/matches/stats").json()["total"],
              len(client.get("/api/research/hypotheses").json()))
    with TestClient(app) as fresh:  # new lifespan == process restart for the API layer
        after = (len(fresh.get("/api/models").json()), fresh.get("/api/matches/stats").json()["total"],
                 len(fresh.get("/api/research/hypotheses").json()))
        assert fresh.get("/api/status").json()["totals"]["runs_completed"] >= 3
    assert before == after and before[2] >= 9


def test_speed_unlocks_once_the_human_has_crashed(lab):
    from backend.api.game import Match
    m = Match("human_vs_ai", [lab["ppo"]], seed=5, limit_ticks=400)
    assert m.human_alive
    while m.world.birds[0].alive:  # the human never flaps and falls to the floor
        m.tick(False)
    assert not m.human_alive and m.world.birds[0].death == "floor"
    assert not Match("ai", [lab["ppo"]], seed=5).human_alive


def test_a_wider_view_shows_more_world_without_changing_the_flight(client, lab):
    def run(extra):
        with client.websocket_connect("/ws/game") as ws:
            ws.send_json({"type": "start", "mode": "ai", "models": [lab["ppo"]], "seed": 12, "speed": 16,
                          "limit_ticks": 200, **extra})
            ready = ws.receive_json()
            states, result = play(ws)
        return ready, states, result

    narrow_ready, narrow, a = run({})
    wide_ready, wide, b = run({"view_width": 1080})
    assert narrow_ready["config"]["view_width"] == 432 and wide_ready["config"]["view_width"] == 1080
    assert max(o["x"] for s in wide for o in s["obstacles"]) > 432 >= max(o["x"] for s in narrow for o in s["obstacles"])
    pick = lambda r: [(p["score"], p["ticks"], p["death"], p["flaps"]) for p in r["players"]]
    assert pick(a) == pick(b)  # same seed, same policy, same flight: the view is not part of the simulation
    assert client.get("/api/game/config").json()["view_width_range"] == [432.0, 1296.0]
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start", "mode": "human", "view_width": 5000})
        assert "view_width must be within" in ws.receive_json()["message"]


def test_websocket_refuses_cross_site_pages(client):
    from starlette.websockets import WebSocketDisconnect
    for origin in ("http://evil.example", "https://localhost.evil.example", "http://127.0.0.1.evil.example:8000"):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/game", headers={"origin": origin}):
                pass
    for origin in ("http://localhost:5173", "http://127.0.0.1:8000", "http://testserver"):  # dev server, served app, same origin
        with client.websocket_connect("/ws/game", headers={"origin": origin}) as ws:
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "error"  # connected: the server answers (no game in progress)
