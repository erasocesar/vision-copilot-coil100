"""Política de decisión determinista: qué recibe el robot.

PICK solo si la imagen es válida, el objeto es conocido y la puntuación de agarre
(probabilidad estimada de clase correcta y ángulo dentro de ±10°) supera el umbral.
Todo lo demás es DIVERT: el objeto va a la bandeja de revisión manual.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .model import Prediction
from .quality import QualityReport


def pick_features(prob, ood, angle_std, sym, ood_scale: float) -> np.ndarray:
    """Entradas de la puntuación de agarre (la misma fórmula en experimentos y en producción)."""
    prob, ood, angle_std, sym = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (prob, ood, angle_std, sym))
    return np.column_stack([prob, ood / ood_scale, np.where(sym == 0, 0.0, angle_std) / 10.0,
                            (sym == 0).astype(float), (sym == 180).astype(float)])


def pick_probability(prob, ood, angle_std, sym, pick_model: dict) -> np.ndarray:
    z = pick_features(prob, ood, angle_std, sym, pick_model["ood_scale"]) @ np.array(pick_model["coef"]) \
        + pick_model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Thresholds:
    max_ood: float
    min_pick_score: float
    pick_model: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Thresholds":
        return cls(d["max_ood"], d["min_pick_score"], d["pick_model"])


@dataclass
class Decision:
    action: Literal["PICK", "DIVERT"]
    obj_id: int | None
    angle: float | None
    angle_mode: Literal["exact", "mod180", "any"]
    pick_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def decide(pred: Prediction, quality: QualityReport, th: Thresholds) -> Decision:
    mode = {0: "any", 180: "mod180"}.get(pred.symmetry, "exact")
    angle = 0.0 if mode == "any" else (pred.angle % 180 if mode == "mod180" else pred.angle)
    score = float(pick_probability(pred.class_prob, pred.ood_distance, pred.angle_std,
                                   pred.symmetry, th.pick_model)[0])
    reasons = list(quality.reasons)
    if not reasons and pred.ood_distance > th.max_ood:
        reasons.append(f"objeto desconocido (distancia {pred.ood_distance:.1f} > {th.max_ood:.1f})")
    if not reasons and score < th.min_pick_score:
        if pred.class_prob < 0.9:
            reasons.append(f"confianza de clase baja ({pred.class_prob:.2f})")
        if pred.symmetry != 0 and pred.angle_std > 10:
            reasons.append(f"ángulo incierto (±{pred.angle_std:.0f}°)")
        reasons.append(f"puntuación de agarre {score:.2f} < {th.min_pick_score:.2f}")
    action = "DIVERT" if reasons else "PICK"
    return Decision(action, pred.obj_id, round(float(angle), 1) if angle is not None else None, mode, score, reasons)
