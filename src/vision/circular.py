"""Estadística circular en grados. El ángulo de COIL-100 da la vuelta en 360°."""
import numpy as np


def circ_dist(a, b):
    """Distancia angular mínima en grados, en [0, 180]."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360
    return np.minimum(d, 360 - d)


def circ_mean(angles, weights=None) -> float:
    """Media circular en [0, 360)."""
    r = np.radians(np.asarray(angles, dtype=float))
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)
    m = np.degrees(np.arctan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())) % 360
    return float(0.0 if np.isclose(m, 360) else m)


def circ_std(angles, weights=None) -> float:
    """Desviación circular, sqrt(-2 ln R), en grados."""
    r = np.radians(np.asarray(angles, dtype=float))
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)
    R = np.hypot((w * np.sin(r)).sum(), (w * np.cos(r)).sum()) / w.sum()
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def sym_dist(a, b, period: int):
    """Error angular teniendo en cuenta la simetría del objeto.

    period=360: objeto asimétrico; 180: se repite cada 180°; 0: simétrico total.
    """
    if period == 0:
        return np.zeros_like(np.asarray(a, dtype=float)) if np.ndim(a) else 0.0
    d = circ_dist(a, b)
    if period == 180:
        return np.minimum(d, 180 - d)
    return d
