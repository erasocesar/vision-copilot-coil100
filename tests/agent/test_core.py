import json

import httpx
import pytest

from src.agent import approvals
from src.agent.audit import audit
from src.agent.context import ToolContext
from src.agent.vision_client import VisionClient, VisionError
from src.common.db import connect, init_schema, rows


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_schema(c)
    c.execute("INSERT INTO line_state VALUES ('P1','L1',0,0.97,'1.0.0')")
    c.commit()
    return c


def ctx(conn, user="ana", role="supervisor"):
    return ToolContext(user, role, "P1", "L1", "s1", conn, vision=None)


# ---------- cliente de visión ----------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _client(handler, clock=None):
    transport = httpx.MockTransport(handler)
    return VisionClient("http://vision", timeout_s=0.5, transport=transport, sleep=lambda s: None,
                        clock=clock or Clock())


def test_retries_then_server_error(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503, json={"error": "x"})

    img = tmp_path / "a.png"
    img.write_bytes(b"png")
    with pytest.raises(VisionError) as e:
        _client(handler).predict(img)
    assert e.value.kind == "server" and len(calls) == 3  # 1 intento + 2 reintentos


def test_timeout(tmp_path):
    def handler(req):
        raise httpx.ReadTimeout("lento", request=req)

    img = tmp_path / "a.png"
    img.write_bytes(b"png")
    with pytest.raises(VisionError) as e:
        _client(handler).predict(img)
    assert e.value.kind == "timeout"


def test_bad_request_not_retried(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": "invalid_image", "detail": "no es imagen"})

    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    with pytest.raises(VisionError) as e:
        _client(handler).predict(img)
    assert e.value.kind == "bad_request" and len(calls) == 1


def test_circuit_breaker(tmp_path):
    calls = []
    clock = Clock()

    def handler(req):
        calls.append(1)
        return httpx.Response(503)

    c = _client(handler, clock)
    img = tmp_path / "a.png"
    img.write_bytes(b"png")
    for _ in range(3):
        with pytest.raises(VisionError):
            c.predict(img)
    n = len(calls)
    with pytest.raises(VisionError) as e:
        c.predict(img)
    assert e.value.kind == "circuit_open" and len(calls) == n
    clock.t = 31
    with pytest.raises(VisionError):
        c.predict(img)
    assert len(calls) > n


# ---------- auditoría ----------

def test_audit_hashes_args(conn):
    audit(ctx(conn), "query_line_metrics", {"shift": 2}, "ok")
    r = rows(conn, "SELECT * FROM audit")[0]
    assert r["tool"] == "query_line_metrics" and len(r["args_sha256"]) == 64 and "shift" not in r["args_sha256"]


# ---------- aprobaciones ----------

def test_request_does_not_execute(conn):
    aid = approvals.create_request(ctx(conn, role="quality"), "set_threshold", {"min_pick_score": 0.9}, "muchos desvíos")
    assert rows(conn, "SELECT min_pick_score FROM line_state")[0]["min_pick_score"] == 0.97
    assert rows(conn, "SELECT status FROM approvals WHERE approval_id=?", (aid,))[0]["status"] == "pending"


def test_requester_role_checked(conn):
    with pytest.raises(PermissionError):
        approvals.create_request(ctx(conn, role="maintenance"), "promote_model", {"version": "1.1.0-rc1"}, "x")


def test_self_approval_rejected(conn):
    aid = approvals.create_request(ctx(conn, "ana", "supervisor"), "pause_line", {}, "limpieza")
    with pytest.raises(PermissionError):
        approvals.decide(conn, aid, ctx(conn, "ana", "plant_manager"), approve=True)


def test_approver_role_checked(conn):
    aid = approvals.create_request(ctx(conn, "ana", "supervisor"), "pause_line", {}, "limpieza")
    with pytest.raises(PermissionError):
        approvals.decide(conn, aid, ctx(conn, "luis", "maintenance"), approve=True)


def test_double_approval_no_double_effect(conn):
    aid = approvals.create_request(ctx(conn, "ana", "quality"), "set_threshold", {"min_pick_score": 0.9}, "x")
    approvals.decide(conn, aid, ctx(conn, "jefe", "supervisor"), approve=True)
    assert rows(conn, "SELECT min_pick_score FROM line_state")[0]["min_pick_score"] == 0.9
    with pytest.raises(ValueError):
        approvals.decide(conn, aid, ctx(conn, "jefe", "supervisor"), approve=True)


def test_manual_angle_written_by_human(conn):
    aid = approvals.create_request(ctx(conn, "ana", "supervisor"), "send_manual_angle",
                                   {"inspection_id": 7, "obj_id": 3, "angle": 45}, "caso dudoso")
    approvals.decide(conn, aid, ctx(conn, "jefe", "plant_manager"), approve=True, manual_angle=50.0)
    cmd = rows(conn, "SELECT * FROM robot_commands WHERE approval_id=?", (aid,))[0]
    assert cmd["angle"] == 50.0 and cmd["source"] == "manual" and cmd["action"] == "PICK"


def test_threshold_out_of_range(conn):
    with pytest.raises(ValueError):
        approvals.create_request(ctx(conn, role="quality"), "set_threshold", {"min_pick_score": 0.2}, "x")


def test_reject_has_no_effect(conn):
    aid = approvals.create_request(ctx(conn, "ana", "supervisor"), "pause_line", {}, "x")
    out = approvals.decide(conn, aid, ctx(conn, "jefe", "plant_manager"), approve=False)
    assert out["status"] == "rejected" and rows(conn, "SELECT paused FROM line_state")[0]["paused"] == 0
    assert json.loads(rows(conn, "SELECT params_json FROM approvals")[0]["params_json"]) == {}
