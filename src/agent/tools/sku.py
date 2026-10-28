"""Alta de producto nuevo (SKU): registro en área aislada y evaluación sin tocar producción."""
import json
import uuid
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from src.common.db import now_iso, rows

from . import ToolError, register
from .image import image_path

MAX_GAP_DEG = 15
EVALUABLE_STATES = ("registered", "evaluated_ok", "evaluated_fail")


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    image_ids: list[Annotated[str, Field(pattern=r"^[0-9a-f]{12,64}$")]] = Field(min_length=6, max_length=72)
    angles: list[int] = Field(min_length=6, max_length=72)

    @model_validator(mode="after")
    def _same_length(self):
        if len(self.image_ids) != len(self.angles):
            raise ValueError("image_ids y angles deben tener la misma longitud")
        if any(not 0 <= x < 360 for x in self.angles):
            raise ValueError("los ángulos van de 0 a 359")
        return self


def _max_gap(angles: list[int]) -> int:
    a = sorted(set(x % 360 for x in angles))
    if len(a) < 2:
        return 360
    return max([b - x for x, b in zip(a, a[1:])] + [a[0] + 360 - a[-1]])


@register("register_sku_candidate",
          "Registra un producto nuevo con sus vistas (image_ids y ángulos) en un área aislada y valida que el "
          "material sea suficiente (hueco máximo entre vistas de 15°).", RegisterIn,
          roles={"ml_engineer", "quality"}, side_effect=True)
def register_sku_candidate(ctx, a: RegisterIn) -> dict:
    for iid in a.image_ids:
        image_path(ctx, iid)
    gap = _max_gap(a.angles)
    status = "registered" if gap <= MAX_GAP_DEG else "insufficient"
    cid = "SKU-" + uuid.uuid4().hex[:8].upper()
    ctx.conn.execute("INSERT INTO sku_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (cid, ctx.plant_id, ctx.line_id, a.name, json.dumps(a.image_ids), json.dumps(a.angles),
                      len(a.angles), gap, status, None, ctx.user_id, now_iso()))
    ctx.conn.commit()
    out = {"candidate_id": cid, "status": status, "n_views": len(a.angles), "max_gap_deg": gap}
    if status == "insufficient":
        out["detail"] = f"hay un hueco de {gap}° entre vistas; hacen falta vistas como máximo cada {MAX_GAP_DEG}°"
    return out


class EvaluateIn(BaseModel):
    candidate_id: str = Field(pattern=r"^SKU-[0-9A-F]{8}$")


@register("evaluate_sku_candidate",
          "Evalúa un candidato registrado en entorno aislado: precisión de ángulo dejando una vista fuera y "
          "parecido con productos del catálogo. No cambia producción.", EvaluateIn,
          roles={"ml_engineer", "quality"})
def evaluate_sku_candidate(ctx, a: EvaluateIn) -> dict:
    c = rows(ctx.conn, "SELECT * FROM sku_candidates WHERE candidate_id=? AND plant_id=? AND line_id=?",
             (a.candidate_id, ctx.plant_id, ctx.line_id))
    if not c:
        raise ToolError("not_found", "no existe ese candidato en esta línea")
    c = c[0]
    if c["status"] == "insufficient":
        raise ToolError("insufficient_views", f"el candidato tiene un hueco de {c['max_gap_deg']}°; registra más vistas")
    # 'production' solo lo fija una aprobación (approvals.execute); esta tool no puede reescribirlo.
    if c["status"] not in EVALUABLE_STATES:
        raise ToolError("invalid_state", f"el candidato está en estado {c['status']} y no se puede evaluar")
    ids, angles = json.loads(c["image_ids"]), json.loads(c["angles"])
    metrics = ctx.vision.evaluate_views([image_path(ctx, i) for i in ids], angles)
    status = "evaluated_ok" if metrics["meets_criteria"] else "evaluated_fail"
    # La evaluación tarda: si mientras tanto se promovió el candidato, este UPDATE no debe pisar 'production'.
    placeholders = ",".join("?" * len(EVALUABLE_STATES))
    cur = ctx.conn.execute(
        "UPDATE sku_candidates SET status=?, metrics_json=? WHERE candidate_id=? AND plant_id=? AND line_id=?"
        f" AND status IN ({placeholders})",
        (status, json.dumps(metrics), a.candidate_id, ctx.plant_id, ctx.line_id, *EVALUABLE_STATES))
    if cur.rowcount != 1:
        raise ToolError("invalid_state", "el estado del candidato cambió durante la evaluación; no se ha modificado")
    ctx.conn.commit()
    return {"candidate_id": a.candidate_id, "status": status, **metrics,
            "next_step": "pedir promote_sku con request_risk_operation" if status == "evaluated_ok" else
            "no puede pasar a producción con este material"}
