"""Detección de simetría de rotación de un objeto a partir de sus vistas."""
import numpy as np


def _mean_dist_at(features: np.ndarray, angles: np.ndarray, delta: int) -> float | None:
    """Distancia media entre vistas separadas `delta` grados.

    Si no hay parejas exactas (p. ej. vistas cada 20° y delta=90), usa la separación
    disponible más cercana a delta.
    """
    pos = {int(a) % 360: i for i, a in enumerate(angles)}
    offsets = sorted({(b - a) % 360 for a in pos for b in pos if a != b},
                     key=lambda o: (min(abs(o - delta), 360 - abs(o - delta)), o))
    if not offsets:
        return None
    best = offsets[0]
    d = [np.linalg.norm(features[i] - features[pos[(a + best) % 360]])
         for a, i in pos.items() if (a + best) % 360 in pos]
    return float(np.mean(d)) if d else None


def rotation_distances(features: np.ndarray, angles: np.ndarray) -> tuple[float | None, float | None]:
    """Distancia media entre vistas separadas 90° y 180°."""
    angles = np.asarray(angles) % 360
    return _mean_dist_at(features, angles, 90), _mean_dist_at(features, angles, 180)


def symmetry_period(features: np.ndarray, angles: np.ndarray, ref_d90: float,
                    full_ratio: float = 0.30, half_ratio: float = 0.40) -> int:
    """Devuelve 0 (simétrico total), 180 o 360 (asimétrico).

    No depende de la separación entre vistas (con vistas cada 30° también vale):
    - simétrico total: girarlo 90° le cambia menos de full_ratio veces lo que cambia
      el objeto mediano del catálogo (ref_d90);
    - simétrico de 180°: girarlo 180° le cambia menos de half_ratio veces lo que le cambia girarlo 90°.
    Umbrales fijados con el catálogo completo e inspección visual (docs/img/simetria_candidatos.png).
    """
    d90, d180 = rotation_distances(features, angles)
    if d90 is None or d180 is None or ref_d90 <= 0:
        return 360
    if d90 <= full_ratio * ref_d90:
        return 0
    if d180 <= half_ratio * d90:
        return 180
    return 360
