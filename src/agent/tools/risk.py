"""Operaciones de riesgo: la tool solo crea la petición; la ejecución exige aprobación humana."""
from typing import Literal

from pydantic import BaseModel, Field

from .. import approvals
from . import ALL, register


class RiskIn(BaseModel):
    op: Literal["set_threshold", "pause_line", "resume_line", "send_manual_angle", "promote_model", "promote_sku"]
    params: dict = Field(default_factory=dict, description=(
        "set_threshold: {min_pick_score}; send_manual_angle: {inspection_id, obj_id, angle}; "
        "promote_model: {version}; promote_sku: {candidate_id}; pause_line y resume_line: {}"))
    justification: str = Field(min_length=5, max_length=500)


@register("request_risk_operation",
          "Pide una operación de riesgo (cambiar el umbral, pausar o reanudar la línea, enviar un ángulo manual "
          "al robot, promover un modelo o un producto). Nunca la ejecuta: queda pendiente de aprobación de "
          "otra persona con rol autorizado.", RiskIn, roles=ALL, side_effect=True)
def request_risk_operation(ctx, a: RiskIn) -> dict:
    aid = approvals.create_request(ctx, a.op, a.params, a.justification)
    return {"approval_id": aid, "status": "pending_approval", "op": a.op,
            "approvers": sorted(approvals.APPROVERS[a.op]),
            "detail": "la operación NO se ha ejecutado; espera la aprobación humana"}
