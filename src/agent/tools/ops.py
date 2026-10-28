"""Consulta operativa: métricas de línea, salud de cámara y reporte de entrega de turno."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.common.db import rows

from . import register

SHIFT_HOURS = {1: (6, 14), 2: (14, 22), 3: (22, 6)}


def hour_filter(start_hour: int | None, end_hour: int | None) -> tuple[str, list]:
    """Filtro por hora del día; admite intervalos que cruzan la medianoche (22 a 6)."""
    if start_hour is None and end_hour is None:
        return "", []
    s = 0 if start_hour is None else start_hour
    e = 24 if end_hour is None else end_hour
    h = "CAST(substr(ts,12,2) AS INTEGER)"
    if s < e:
        return f" AND {h} >= ? AND {h} < ?", [s, e]
    return f" AND ({h} >= ? OR {h} < ?)", [s, e]


class MetricsIn(BaseModel):
    shift: Literal[1, 2, 3] | None = Field(None, description="Turno: 1 (06-14), 2 (14-22), 3 (22-06)")
    start_hour: int | None = Field(None, ge=0, le=23)
    end_hour: int | None = Field(None, ge=0, le=24)
    group_by: Literal["hour", "shift", "model_version"] = "hour"

    @model_validator(mode="after")
    def _shift_or_hours(self):
        if self.shift is not None and (self.start_hour is not None or self.end_hour is not None):
            raise ValueError("usa shift o start_hour/end_hour, no ambos")
        return self


@register("query_line_metrics",
          "Métricas de la línea en las últimas 24 h: piezas, PICK, DIVERT, tasa de desvío, confianza media y "
          "desenfoque medio, agrupadas por hora, turno o versión de modelo.",
          MetricsIn)
def query_line_metrics(ctx, a: MetricsIn) -> dict:
    where, params = " WHERE plant_id=? AND line_id=?", [ctx.plant_id, ctx.line_id]
    if a.shift is not None:
        f, p = hour_filter(*SHIFT_HOURS[a.shift])
    else:
        f, p = hour_filter(a.start_hour, a.end_hour)
    bucket = {"hour": "substr(ts,12,2)", "shift": "shift", "model_version": "model_version"}[a.group_by]
    sql = (f"SELECT {bucket} AS bucket, COUNT(*) AS total, SUM(action='PICK') AS picks, SUM(action='DIVERT') AS diverts,"
           f" ROUND(AVG(action='DIVERT'),3) AS divert_rate, ROUND(AVG(class_prob),3) AS avg_class_prob,"
           f" ROUND(AVG(camera_blur),1) AS avg_blur FROM inspections{where}{f} GROUP BY bucket ORDER BY MIN(ts)")
    data = rows(ctx.conn, sql, tuple(params + p))
    tot = sum(r["total"] for r in data)
    div = sum(r["diverts"] for r in data)
    return {"rows": data, "total": tot, "divert_rate": round(div / tot, 3) if tot else None}


class CameraIn(BaseModel):
    start_hour: int | None = Field(None, ge=0, le=23)
    end_hour: int | None = Field(None, ge=0, le=24)


@register("get_camera_health",
          "Salud de la cámara por hora: desenfoque (varianza del laplaciano, más bajo = peor), brillo, "
          "fotogramas perdidos y temperatura.", CameraIn)
def get_camera_health(ctx, a: CameraIn) -> dict:
    f, p = hour_filter(a.start_hour, a.end_hour)
    data = rows(ctx.conn, "SELECT substr(ts,12,2) AS hour, ROUND(AVG(blur),1) AS blur, ROUND(AVG(brightness),1) AS"
                          " brightness, SUM(dropped_frames) AS dropped_frames, ROUND(AVG(temperature_c),1) AS"
                          f" temperature_c FROM camera_health WHERE plant_id=? AND line_id=?{f}"
                          " GROUP BY hour ORDER BY MIN(ts)", tuple([ctx.plant_id, ctx.line_id] + p))
    base = sorted(r["blur"] for r in data)
    normal = base[len(base) // 2] if base else None
    return {"rows": data, "median_blur": normal,
            "note": "un desenfoque muy por debajo de la mediana indica pérdida de foco"}


class ShiftReportIn(BaseModel):
    shift: Literal[1, 2, 3]


@register("generate_shift_report",
          "Reporte de entrega de turno: métricas, incidentes detectados, tickets abiertos, casos pendientes y "
          "operaciones de riesgo pendientes de aprobación.", ShiftReportIn)
def generate_shift_report(ctx, a: ShiftReportIn) -> dict:
    f, p = hour_filter(*SHIFT_HOURS[a.shift])
    base = (ctx.plant_id, ctx.line_id)
    by_hour = rows(ctx.conn, "SELECT substr(ts,12,2) AS hour, COUNT(*) AS total, ROUND(AVG(action='DIVERT'),3) AS"
                             f" divert_rate FROM inspections WHERE plant_id=? AND line_id=?{f} GROUP BY hour",
                   tuple(list(base) + p))
    tot = sum(r["total"] for r in by_hour)
    avg = sum(r["divert_rate"] * r["total"] for r in by_hour) / tot if tot else 0
    incidents = [f"tasa de desvío {r['divert_rate']:.0%} a las {r['hour']}:00 (media del turno {avg:.0%})"
                 for r in by_hour if r["divert_rate"] > max(2 * avg, 0.3)]
    grasp = rows(ctx.conn, "SELECT failure_code, COUNT(*) AS n FROM grasp_events WHERE plant_id=? AND line_id=?"
                           f" AND success=0{f} GROUP BY failure_code", tuple(list(base) + p))
    tickets = rows(ctx.conn, "SELECT ticket_id, priority FROM tickets WHERE plant_id=? AND line_id=?"
                             " AND status='open'", base)
    pending = rows(ctx.conn, "SELECT COUNT(*) AS n FROM review_queue q JOIN inspections i ON i.id=q.inspection_id"
                             f" WHERE q.plant_id=? AND q.line_id=? AND q.status='pending'{f.replace('ts', 'i.ts')}",
                   tuple(list(base) + p))[0]["n"]
    approvals = rows(ctx.conn, "SELECT approval_id, op, requested_by FROM approvals WHERE plant_id=? AND line_id=?"
                               " AND status='pending'", base)
    state = rows(ctx.conn, "SELECT paused, min_pick_score, model_version FROM line_state WHERE plant_id=?"
                           " AND line_id=?", base)
    return {"shift": a.shift, "total": tot, "divert_rate": round(avg, 3), "incidents": incidents,
            "grasp_failures": {g["failure_code"]: g["n"] for g in grasp}, "open_tickets": [{"ticket_id": t["ticket_id"], "priority": t["priority"]} for t in tickets],
            "pending_review_cases": pending, "pending_approvals": approvals,
            "line_state": state[0] if state else None}

