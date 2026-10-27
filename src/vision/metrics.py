"""Métricas de evaluación: tolerancia angular, calibración, riesgo-cobertura y rechazo."""
import numpy as np


def within_tol(err_deg, tol: float = 10.0) -> float:
    err = np.asarray(err_deg, dtype=float)
    return float((err <= tol).mean()) if err.size else float("nan")


def ece(conf, correct, n_bins: int = 15) -> float:
    """Error de calibración esperado: diferencia media entre confianza y acierto por tramo."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def risk_coverage(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    """Al aceptar las predicciones de mayor confianza, qué fracción se cubre y qué fracción falla."""
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    errors = 1 - np.asarray(correct, dtype=float)[order]
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(errors) / n


def choose_threshold(conf, correct, max_risk: float = 0.01, weights=None) -> float:
    """Menor umbral de confianza con riesgo entre aceptados <= max_risk (máxima cobertura).

    `weights` permite reponderar muestras (p. ej. ajustar la proporción de objetos desconocidos
    a la esperada en planta en lugar de la del conjunto de calibración).
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, dtype=float)
    best = np.inf
    for t in np.unique(conf)[::-1]:
        acc = conf >= t
        if (w[acc] * (1 - correct[acc])).sum() / w[acc].sum() <= max_risk:
            best = float(t)
    return best


def auroc(score_known, score_unknown) -> float:
    """Probabilidad de que un conocido puntúe más alto que un desconocido (empates cuentan 0,5)."""
    k = np.asarray(score_known, dtype=float)[:, None]
    u = np.asarray(score_unknown, dtype=float)[None, :]
    return float(((k > u).sum() + 0.5 * (k == u).sum()) / (k.size * u.size))


def pick_outcome(pred_obj, true_obj, err_deg, accepted, tol: float = 10.0) -> dict:
    """Cifras de negocio: cuánto agarra el robot solo y cuántos de esos agarres salen mal."""
    pred_obj, true_obj = np.asarray(pred_obj), np.asarray(true_obj)
    err = np.asarray(err_deg, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    cls_ok = pred_obj == true_obj
    good_pick = cls_ok & (err <= tol)
    n_acc = accepted.sum()
    return {
        "coverage": float(accepted.mean()),
        "wrong_pick_rate": float((~good_pick[accepted]).mean()) if n_acc else 0.0,
        "class_acc": float(cls_ok.mean()),
        "angle_within_tol": float((err[cls_ok] <= tol).mean()) if cls_ok.any() else float("nan"),
    }
