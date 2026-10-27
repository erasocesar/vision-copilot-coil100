"""Operaciones de riesgo: el agente solo puede pedirlas; las ejecuta código tras la aprobación de otra persona."""
import json
import uuid

from pydantic import BaseModel, Field, ValidationError

from src.common.db import now_iso, rows

from .audit import audit


class SetThreshold(BaseModel):
    min_pick_score: float = Field(ge=0.5, le=0.999)


class NoParams(BaseModel):
    pass


class ManualAngle(BaseModel):
    inspection_id: int = Field(ge=1)
    obj_id: int = Field(ge=1, le=100)
    angle: float = Field(ge=0, lt=360)


class PromoteModel(BaseModel):
    version: str = Field(min_length=1, max_length=40)


class PromoteSku(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=64)


# operación: (parámetros, roles que pueden pedirla, roles que pueden aprobarla)
OPS = {
    "set_threshold": (SetThreshold, {"quality", "ml_engineer", "supervisor"}, {"supervisor", "plant_manager"}),
    "pause_line": (NoParams, {"supervisor", "maintenance", "plant_manager"}, {"supervisor", "plant_manager"}),
    "resume_line": (NoParams, {"supervisor", "maintenance", "plant_manager"}, {"supervisor", "plant_manager"}),
    "send_manual_angle": (ManualAngle, {"supervisor", "quality"}, {"supervisor", "plant_manager"}),
    "promote_model": (PromoteModel, {"ml_engineer"}, {"plant_manager"}),
    "promote_sku": (PromoteSku, {"ml_engineer", "quality"}, {"plant_manager"}),
}
REQUESTERS = {op: v[1] for op, v in OPS.items()}
APPROVERS = {op: v[2] for op, v in OPS.items()}


def create_request(ctx, op: str, params: dict, justification: str) -> str:
    if op not in OPS:
        raise ValueError(f"operación desconocida: {op}")
    model, requesters, _ = OPS[op]
    if ctx.role not in requesters:
        raise PermissionError(f"el rol {ctx.role} no puede pedir {op}")
    try:
        clean = model(**(params or {})).model_dump()
    except ValidationError as e:
        raise ValueError(f"parámetros inválidos para {op}: {e.errors()[0]['msg']}") from None
    aid = uuid.uuid4().hex[:12]
    ctx.conn.execute(
        "INSERT INTO approvals (approval_id, plant_id, line_id, op, params_json, justification, requested_by,"
        " requested_role, ts_requested) VALUES (?,?,?,?,?,?,?,?,?)",
        (aid, ctx.plant_id, ctx.line_id, op, json.dumps(clean), justification[:500], ctx.user_id, ctx.role, now_iso()))
    ctx.conn.commit()
    return aid


def get(conn, approval_id: str, plant_id: str, line_id: str) -> dict | None:
    r = rows(conn, "SELECT * FROM approvals WHERE approval_id=? AND plant_id=? AND line_id=?",
             (approval_id, plant_id, line_id))
    return r[0] if r else None


def decide(conn, approval_id: str, approver, approve: bool, manual_angle: float | None = None) -> dict:
    a = get(conn, approval_id, approver.plant_id, approver.line_id)
    if a is None:
        raise ValueError("aprobación no encontrada")
    if a["status"] != "pending":
        raise ValueError(f"la aprobación ya está en estado {a['status']}")
    if approver.user_id == a["requested_by"]:
        raise PermissionError("nadie puede aprobar su propia petición")
    if approver.role not in APPROVERS[a["op"]]:
        raise PermissionError(f"el rol {approver.role} no puede aprobar {a['op']}")
    params = json.loads(a["params_json"])
    if a["op"] == "send_manual_angle" and approve:
        if manual_angle is None:
            raise ValueError("el ángulo manual lo escribe quien aprueba")
        params["angle"] = ManualAngle(**{**params, "angle": manual_angle}).angle
    status = "approved" if approve else "rejected"
    # Transición atómica: solo una decisión gana aunque lleguen dos a la vez.
    cur = conn.execute("UPDATE approvals SET status=?, approved_by=?, approved_role=?, ts_decided=?"
                       " WHERE approval_id=? AND status='pending'",
                       (status, approver.user_id, approver.role, now_iso(), approval_id))
    if cur.rowcount != 1:
        conn.rollback()
        raise ValueError("la aprobación ya fue decidida")
    result = execute(conn, {**a, "params": params}) if approve else {"executed": False}
    conn.execute("UPDATE approvals SET result_json=? WHERE approval_id=?", (json.dumps(result), approval_id))
    conn.commit()
    audit(approver, f"approval:{a['op']}", {"approval_id": approval_id, "approve": approve}, status)
    return {"approval_id": approval_id, "status": status, "result": result}


def execute(conn, a: dict) -> dict:
    op, p, plant, line = a["op"], a["params"], a["plant_id"], a["line_id"]
    if op == "set_threshold":
        conn.execute("UPDATE line_state SET min_pick_score=? WHERE plant_id=? AND line_id=?",
                     (p["min_pick_score"], plant, line))
    elif op in ("pause_line", "resume_line"):
        conn.execute("UPDATE line_state SET paused=? WHERE plant_id=? AND line_id=?",
                     (int(op == "pause_line"), plant, line))
    elif op == "send_manual_angle":
        conn.execute("INSERT INTO robot_commands (plant_id, line_id, ts, inspection_id, action, obj_id, angle,"
                     " angle_mode, source, approval_id) VALUES (?,?,?,?,'PICK',?,?,'exact','manual',?)",
                     (plant, line, now_iso(), p["inspection_id"], p["obj_id"], p["angle"], a["approval_id"]))
    elif op == "promote_model":
        # El registro de versiones es global; la promoción solo cambia la versión de esta planta y línea.
        if not rows(conn, "SELECT 1 FROM model_versions WHERE version=? AND stage IN ('staging','production')",
                    (p["version"],)):
            return {"executed": False, "error": "la versión no está registrada"}
        conn.execute("UPDATE line_state SET model_version=? WHERE plant_id=? AND line_id=?",
                     (p["version"], plant, line))
    elif op == "promote_sku":
        r = rows(conn, "SELECT status FROM sku_candidates WHERE candidate_id=? AND plant_id=?",
                 (p["candidate_id"], plant))
        if not r or r[0]["status"] != "evaluated_ok":
            return {"executed": False, "error": "el candidato no ha pasado la evaluación"}
        conn.execute("UPDATE sku_candidates SET status='production' WHERE candidate_id=?", (p["candidate_id"],))
    return {"executed": True, "op": op, "params": p}
