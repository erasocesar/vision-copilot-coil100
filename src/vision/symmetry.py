"""Detección de simetría de rotación de un objeto a partir de sus vistas."""
import numpy as np


def _mean_dist_at(features: np.ndarray, angles: np.ndarray, delta: int) -> float | None:
    pos = {int(a) % 360: i for i, a in enumerate(angles)}
    d = [np.linalg.norm(features[i] - features[pos[(a + delta) % 360]])
         for a, i in pos.items() if (a + delta) % 360 in pos]
    return float(np.mean(d)) if d else None


def symmetry_period(features: np.ndarray, angles: np.ndarray,
                    full_ratio: float = 1.5, half_ratio: float = 0.5) -> int:
    """Devuelve 0 (simétrico total), 180 o 360 (asimétrico).

    Compara la distancia media entre vistas separadas un paso, 90° y 180°.
    Simétrico total: girar 90° o 180° cambia tan poco como girar un paso.
    Simétrico de 180°: las vistas opuestas se parecen mucho más que las separadas 90°.
    """
    angles = np.asarray(angles) % 360
    uniq = np.unique(angles)
    step = int(np.min(np.diff(uniq))) if len(uniq) > 1 else 5
    d_step = _mean_dist_at(features, angles, step)
    d90 = _mean_dist_at(features, angles, 90)
    d180 = _mean_dist_at(features, angles, 180)
    if d_step is None or d90 is None or d180 is None:
        return 360
    if d90 <= full_ratio * d_step and d180 <= full_ratio * d_step:
        return 0
    if d180 <= half_ratio * d90:
        return 180
    return 360
