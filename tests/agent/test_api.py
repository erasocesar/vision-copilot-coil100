import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agent.api import MAX_MSGS_PER_HOUR, create_app
from src.common.config import Settings
from src.common.db import connect, init_schema
from src.sim.line_sim import seed_state

from .conftest import IMG_ID, FakeVision
from .test_graph import ScriptedLLM, call


@pytest.fixture
def app_env(tmp_path, image_dir):
    db = tmp_path / "line.db"
    c = connect(db)
    init_schema(c)
    seed_state(c, "P1", "L1", 0.97, "1.0.0")
    c.close()
    vision = FakeVision()
    vision.http = httpx.Client(base_url="http://vision", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"image_id": IMG_ID, "quality": {"ok": True, "reasons": []}})))
    llm = ScriptedLLM([])
    s = Settings(db_path=db, image_dir=image_dir)
    app = create_app(s, llm=llm, vision=vision)
    return TestClient(app, raise_server_exceptions=False), llm


def open_session(client, user="ana", role="supervisor"):
    return client.post("/api/session", json={"user_id": user, "role": role}).json()["session_id"]


def test_flow_upload_and_chat(app_env):
    client, llm = app_env
    sid = open_session(client)
    up = client.post("/api/images", headers={"X-Session": sid}, files={"file": ("a.png", b"png", "image/png")})
    assert up.status_code == 200 and up.json()["image_id"] == IMG_ID
    llm.script[:] = [call("analyze_image", {"image_id": IMG_ID}), AIMessage(content="Producto 7")]
    r = client.post("/api/chat", json={"session_id": sid, "message": "¿qué es?", "image_ids": [IMG_ID]})
    assert r.status_code == 200 and r.json()["tool_trace"][0]["tool"] == "analyze_image"


def test_invalid_session(app_env):
    client, _ = app_env
    assert client.post("/api/chat", json={"session_id": "x", "message": "hola"}).status_code == 401


def test_invalid_role(app_env):
    client, _ = app_env
    assert client.post("/api/session", json={"user_id": "ana", "role": "admin"}).status_code == 422


def test_approval_by_other_user_executes(app_env):
    client, llm = app_env
    sid = open_session(client, "ana", "quality")
    llm.script[:] = [call("request_risk_operation", {"op": "set_threshold", "params": {"min_pick_score": 0.9},
                                                     "justification": "muchos desvíos"}),
                     AIMessage(content="Pedido")]
    pid = client.post("/api/chat", json={"session_id": sid, "message": "baja el umbral"}).json()["pending_approval"]
    assert pid
    own = client.post(f"/api/approvals/{pid}/decision", json={"session_id": sid, "approve": True})
    assert own.status_code == 403
    boss = open_session(client, "jefe", "supervisor")
    assert client.post(f"/api/approvals/{pid}/decision", json={"session_id": boss, "approve": True}).status_code == 200
    assert client.get(f"/api/line?session_id={boss}").json()["min_pick_score"] == 0.9
    again = client.post(f"/api/approvals/{pid}/decision", json={"session_id": boss, "approve": True})
    assert again.status_code == 409


def test_rate_limit(app_env):
    client, llm = app_env
    sid = open_session(client)
    codes = [client.post("/api/chat", json={"session_id": sid, "message": "hola"}).status_code
             for _ in range(MAX_MSGS_PER_HOUR + 1)]
    assert codes[-1] == 429 and codes[0] == 200


def test_errors_hide_traceback(app_env, monkeypatch):
    client, _ = app_env
    import src.agent.api as api
    sid = open_session(client)
    monkeypatch.setattr(api, "run_turn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("detalle secreto")))
    r = client.post("/api/chat", json={"session_id": sid, "message": "hola"})
    assert r.status_code == 500 and "secreto" not in r.text and "Traceback" not in r.text


def test_index_served(app_env):
    client, _ = app_env
    r = client.get("/")
    assert r.status_code == 200 and "Copiloto de línea" in r.text
