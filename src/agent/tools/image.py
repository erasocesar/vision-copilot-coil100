"""Inspección de imagen: calidad y análisis completo (el modelo del reto 2 es la herramienta)."""
import re
from pathlib import Path

from pydantic import BaseModel, Field

from src.common.db import rows

from . import ToolError, register

IMAGE_ID = Field(pattern=r"^[0-9a-f]{12,64}$",
                 description="Identificador de la imagen (los 12 primeros caracteres del SHA-256 bastan)")
SCORE_REASONS = ("puntuación de agarre", "confianza de clase", "ángulo incierto")


def image_path(ctx, image_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12,64}", image_id or ""):
        raise ToolError("invalid_args", "image_id debe ser hexadecimal (12 a 64 caracteres)")
    base = Path(ctx.image_dir).resolve()
    matches = list(base.glob(f"{image_id}*.png")) if len(image_id) < 64 else [base / f"{image_id}.png"]
    if len(matches) > 1:
        raise ToolError("ambiguous_image_id", "hay varias imágenes con ese prefijo; usa más caracteres")
    p = matches[0].resolve() if matches else base / "missing.png"
    if p.parent != base or not p.exists():
        raise ToolError("image_not_found", "no hay ninguna imagen con ese identificador; pide al usuario que la suba")
    return p


def line_threshold(ctx) -> float:
    r = rows(ctx.conn, "SELECT min_pick_score FROM line_state WHERE plant_id=? AND line_id=?",
             (ctx.plant_id, ctx.line_id))
    return float(r[0]["min_pick_score"]) if r else 0.97


class QualityIn(BaseModel):
    image_id: str = IMAGE_ID


@register("check_image_quality",
          "Valida la calidad de una imagen subida (desenfoque, exposición, objeto aislado) sin ejecutar el modelo.",
          QualityIn)
def check_image_quality(ctx, a: QualityIn) -> dict:
    r = ctx.vision.quality(image_path(ctx, a.image_id))
    q = r["quality"]
    return {"image_id": a.image_id, "ok": q["ok"], "reasons": q["reasons"],
            "blur": round(q["blur"], 1), "brightness": round(q["brightness"], 1)}


class AnalyzeIn(BaseModel):
    image_id: str = IMAGE_ID
    top_k: int = Field(3, ge=1, le=5, description="Número de vistas similares a devolver como evidencia")


@register("analyze_image",
          "Identifica el producto, estima su ángulo (grados 0-360) con incertidumbre, devuelve las vistas más "
          "parecidas como evidencia y la decisión para el robot (PICK o DIVERT) con el umbral vigente de la línea.",
          AnalyzeIn)
def analyze_image(ctx, a: AnalyzeIn) -> dict:
    r = ctx.vision.predict(image_path(ctx, a.image_id), top_k=a.top_k)
    p, d = r["prediction"], r["decision"]
    th = line_threshold(ctx)
    blocking = [x for x in d["reasons"] if not x.startswith(SCORE_REASONS)]
    action = "PICK" if not blocking and d["pick_score"] >= th else "DIVERT"
    reasons = blocking or ([] if action == "PICK" else
                           [x for x in d["reasons"] if x.startswith(SCORE_REASONS)] or
                           [f"puntuación de agarre {d['pick_score']:.2f} < umbral de línea {th:.2f}"])
    return {
        "image_id": a.image_id, "obj_id": p["obj_id"], "class_prob": round(p["class_prob"], 3),
        "angle": d["angle"], "angle_std": round(p["angle_std"], 1), "angle_mode": d["angle_mode"],
        "symmetry": p["symmetry"], "angle_instruction": {
            "any": "objeto simétrico: cualquier giro vale, el robot no necesita compensar el ángulo",
            "mod180": f"objeto simétrico cada 180°: el robot compensa {d['angle']}° (o {d['angle'] + 180:.1f}°, es equivalente)",
            "exact": f"el robot compensa {d['angle']}°"}[d["angle_mode"]],
        "pick_score": round(d["pick_score"], 3), "line_threshold": th,
        "action": action, "reasons": reasons, "quality_ok": r["quality"]["ok"],
        "evidence_topk": [{"obj_id": n["obj_id"], "angle": n["angle"], "distance": round(n["distance"], 1)}
                          for n in p["topk"]],
        "model_version": r["model_version"],
    }
