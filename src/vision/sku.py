"""Evaluación de un producto nuevo (SKU) a partir de sus vistas, en un entorno aislado.

No modifica el modelo en producción: mide si con esas vistas el ángulo sería fiable
y si el producto se confundiría con alguno del catálogo.
"""
import numpy as np

from .circular import circ_mean, sym_dist
from .model import VisionModel, featurize
from .symmetry import symmetry_period

MIN_WITHIN_TOL = 0.80
MAX_GAP_DEG = 15


def max_gap(angles: list[int]) -> int:
    a = sorted(int(x) % 360 for x in angles)
    if len(a) < 2:
        return 360
    gaps = np.diff(a + [a[0] + 360])
    return int(gaps.max())


def evaluate_views(model: VisionModel, images: np.ndarray, angles: list[int], max_ood: float) -> dict:
    angles = np.asarray(angles, dtype=int) % 360
    cfg = model.config
    Za = model.angle_proj.transform(featurize(images, cfg.angle_features))
    sym = symmetry_period(Za, angles, model.ref_d90)
    errs = []
    for i in range(len(Za)):
        others = np.delete(np.arange(len(Za)), i)
        d = np.linalg.norm(Za[others] - Za[i], axis=1)
        nn = others[np.argsort(d)[:2]]
        w = 1.0 / (np.sort(d)[:2] + 1e-6)
        mult = 2 if sym == 180 else 1
        est = 0.0 if sym == 0 else circ_mean(angles[nn] * mult, w) / mult
        errs.append(float(sym_dist(est, angles[i], sym)))
    within = float(np.mean(np.array(errs) <= 10)) if errs else 0.0

    Zc = model.class_proj.transform(featurize(images, cfg.class_features))
    dist, idx = model.class_nn.kneighbors(Zc, n_neighbors=1)
    closest = model.obj_ids[idx[:, 0]]
    conflict_obj = int(np.bincount(closest).argmax())
    median_dist = float(np.median(dist[:, 0]))
    conflict = median_dist <= max_ood
    gap = max_gap(angles.tolist())

    reasons = []
    if gap > MAX_GAP_DEG:
        reasons.append(f"hay un hueco de {gap}° entre vistas (máximo {MAX_GAP_DEG}°)")
    if within < MIN_WITHIN_TOL:
        reasons.append(f"solo el {within:.0%} de las vistas dejadas fuera queda dentro de ±10° (mínimo {MIN_WITHIN_TOL:.0%})")
    if conflict:
        reasons.append(f"se parece demasiado al producto {conflict_obj} del catálogo")
    return {
        "n_views": int(len(angles)), "max_gap_deg": gap, "symmetry": int(sym),
        "loo_within_tol": round(within, 3), "loo_median_err": round(float(np.median(errs)), 2) if errs else None,
        "closest_catalog_obj": conflict_obj, "median_catalog_distance": round(median_dist, 2),
        "catalog_distance_threshold": round(float(max_ood), 2),
        "meets_criteria": not reasons, "reasons": reasons,
    }
