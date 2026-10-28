"""Cola de revisión humana: casos desviados y etiquetas confirmadas por personas."""
import json

from pydantic import BaseModel, Field

from src.common.db import now_iso, rows

from . import ToolError, register


class QueueIn(BaseModel):
    limit: int = Field(5, ge=1, le=20)
    reason_contains: str | None = Field(None, max_length=40, description="Filtra por texto del motivo, p. ej. 'desenfocada'")


@register("list_review_queue",
          "Lista casos pendientes de revisión humana (desviados por baja confianza) con su evidencia y motivo.",
          QueueIn)
def list_review_queue(ctx, a: QueueIn) -> dict:
    sql = ("SELECT q.case_id, i.ts, i.shift, i.image_id, i.obj_id_pred, i.class_prob, i.angle_pred, i.angle_std,"
           " i.pick_score, i.reasons FROM review_queue q JOIN inspections i ON i.id = q.inspection_id"
           " WHERE q.plant_id=? AND q.line_id=? AND q.status='pending'")
    params: list = [ctx.plant_id, ctx.line_id]
    if a.reason_contains:
        sql += " AND i.reasons LIKE ?"
        params.append(f"%{a.reason_contains}%")
    total = rows(ctx.conn, sql.replace("SELECT q.case_id, i.ts, i.shift, i.image_id, i.obj_id_pred, i.class_prob,"
                                       " i.angle_pred, i.angle_std, i.pick_score, i.reasons", "SELECT COUNT(*) AS n"),
                 tuple(params))[0]["n"]
    items = rows(ctx.conn, sql + " ORDER BY i.ts DESC LIMIT ?", tuple(params + [a.limit]))
    for it in items:
        it["reasons"] = json.loads(it["reasons"])
        for k in ("class_prob", "angle_pred", "angle_std", "pick_score"):
            it[k] = None if it[k] is None else round(it[k], 2)
    return {"pending_total": total, "cases": items}


class LabelIn(BaseModel):
    case_id: str = Field(max_length=64)
    obj_id: int = Field(ge=1, le=100, description="Producto confirmado por la persona")
    angle: int = Field(ge=0, le=359, description="Ángulo confirmado por la persona, en grados")


@register("submit_review_label",
          "Registra el producto y el ángulo confirmados por una persona para un caso de la cola y lo añade al "
          "conjunto de reentrenamiento. Idempotente por case_id.",
          LabelIn, roles={"supervisor", "quality"}, side_effect=True)
def submit_review_label(ctx, a: LabelIn) -> dict:
    case = rows(ctx.conn, "SELECT q.case_id, i.image_id FROM review_queue q JOIN inspections i ON i.id=q.inspection_id"
                          " WHERE q.case_id=? AND q.plant_id=? AND q.line_id=?", (a.case_id, ctx.plant_id, ctx.line_id))
    if not case:
        raise ToolError("not_found", "no existe ese caso en esta línea")
    prev = rows(ctx.conn, "SELECT obj_id, angle FROM labels WHERE case_id=?", (a.case_id,))
    if prev:
        if prev[0]["obj_id"] == a.obj_id and prev[0]["angle"] == a.angle:
            return {"case_id": a.case_id, "created": False, "detail": "ya estaba etiquetado con esos valores"}
        raise ToolError("conflict", f"el caso ya tiene otra etiqueta (producto {prev[0]['obj_id']}, {prev[0]['angle']}°)")
    ts = now_iso()
    ctx.conn.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?)",
                     (a.case_id, ctx.plant_id, ctx.line_id, a.obj_id, a.angle, ctx.user_id, ts))
    # Los valores los eligió el LLM, sin confirmación determinista: no se registran como 'human_review'.
    ctx.conn.execute("INSERT OR IGNORE INTO retrain_dataset VALUES (?,?,?,?,?,'agent_proposed',?)",
                     (case[0]["image_id"], ctx.plant_id, ctx.line_id, a.obj_id, a.angle, ts))
    ctx.conn.execute("UPDATE review_queue SET status='labeled' WHERE case_id=?", (a.case_id,))
    ctx.conn.commit()
    return {"case_id": a.case_id, "created": True, "added_to_retrain_dataset": True}
