"""Barrido de experimentos de los protocolos A-D y exportación del modelo final.

Uso:
  python -m src.vision.sweep --stage grid     # clasificadores x características x protocolos, baseline y ResNet-50
  python -m src.vision.sweep --stage angle    # knn frente a kernel ridge para el ángulo, con la mejor configuración
  python -m src.vision.sweep --stage export   # entrena y exporta models/vision/v1.0.0
La selección usa solo métricas de validación (val_*); las de prueba se registran pero no deciden.
"""
import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_images, load_index
from .experiments import run_protocol
from .features import cache_features
from .model import ModelConfig

RAW = Path("data/raw")
CACHE = Path("data/cache")
SUMMARY = Path("eval/vision_summary.csv")
DATA_SHA = "fcb4b8b01c0778f416d8d7a09f45082c4f7075cc74743dcddd5c44307bd4afe7"
PROTOCOLS = ["A", "B", "C", "D"]
CLASS_SETS = [["color", "hog"], ["color", "hog", "lbp"], ["color", "lbp", "pixels"]]
CLASSIFIERS = ["knn", "svm", "rf", "logreg"]

BASELINE = ModelConfig(classifier="knn", class_features=["pixels"], angle_features=["pixels"],
                       pca_dims=0, k_angle=1, calibrate=False)


def load_all():
    index = load_index(RAW)
    images = load_images(index)
    feats = {k: cache_features(images, [k], CACHE / f"{k}.npz", DATA_SHA) for k in ["color", "hog", "lbp", "pixels"]}
    cnn_path = CACHE / "cnn.npy"
    if cnn_path.exists():
        feats["cnn"] = np.load(cnn_path)
    return index, images, feats


def ensure_cnn(images) -> np.ndarray:
    path = CACHE / "cnn.npy"
    if not path.exists():
        from .reference_cnn import embed
        np.save(path, embed(images))
    return np.load(path)


def score(row: dict) -> float:
    """Criterio de selección en validación: cobertura, penalizando agarres erróneos."""
    return row["val_coverage"] - 10 * row["val_wrong_pick_rate"]


def record(res: dict, family: str, name: str) -> dict:
    row = {"family": family, "name": name, "protocol": res["protocol"],
           **{k: v for k, v in asdict(res["config"]).items() if not isinstance(v, list)},
           "class_features": "+".join(res["config"].class_features),
           "angle_features": "+".join(res["config"].angle_features),
           "max_ood": res["thresholds"]["max_ood"], "min_pick_score": res["thresholds"]["min_pick_score"],
           **res["metrics"]}
    row["selection_score"] = score(row)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(SUMMARY, mode="a", header=not SUMMARY.exists(), index=False)
    print(f"[{res['protocol']}] {family}/{name}: val_cov={row['val_coverage']:.3f} "
          f"val_wrong={row['val_wrong_pick_rate']:.4f} test_acc={row['test_class_acc']:.3f} "
          f"test_cov={row['test_coverage']:.3f} test_wrong={row['test_wrong_pick_rate']:.4f}", flush=True)
    return row


def stage_grid(index, images, feats):
    feats["cnn"] = ensure_cnn(images)
    for p in PROTOCOLS:
        record(run_protocol(p, BASELINE, index, feats, DATA_SHA, run_name="baseline"), "baseline", "1nn-pixels")
        for clf in CLASSIFIERS:
            for cs in CLASS_SETS:
                cfg = ModelConfig(classifier=clf, class_features=cs)
                record(run_protocol(p, cfg, index, feats, DATA_SHA), "classical", f"{clf}-{'+'.join(cs)}")
        for clf in ["knn", "logreg"]:
            cfg = ModelConfig(classifier=clf, class_features=["cnn"], angle_features=["cnn"])
            record(run_protocol(p, cfg, index, feats, DATA_SHA, run_name=f"resnet50-{clf}"), "reference", f"resnet50-{clf}")


def best_classical() -> ModelConfig:
    df = pd.read_csv(SUMMARY)
    df = df[df.family == "classical"]
    mean = df.groupby("name")["selection_score"].mean().sort_values(ascending=False)
    best = df[df.name == mean.index[0]].iloc[0]
    print("Mejor configuración clásica (media de validación en A-D):", mean.index[0], round(mean.iloc[0], 4))
    return ModelConfig(classifier=best.classifier, class_features=best.class_features.split("+"),
                       angle_features=best.angle_features.split("+"))


def stage_angle(index, images, feats):
    base = best_classical()
    for p in PROTOCOLS:
        for method, k in [("knn", 1), ("knn", 5), ("kernel_ridge", 3)]:
            cfg = replace(base, angle_method=method, k_angle=k)
            record(run_protocol(p, cfg, index, feats, DATA_SHA, run_name=f"angle-{method}-k{k}"), "angle", f"{method}-k{k}")


def stage_export(index, images, feats, out=Path("models/vision/v1.0.0")):
    """El modelo de la demo se entrena con la partición D: los 10 objetos excluidos siguen siendo
    desconocidos de verdad para la demo y para el agente."""
    df = pd.read_csv(SUMMARY)
    cfg = best_classical()
    ang = df[df.family == "angle"]
    if len(ang):
        best_angle = ang.groupby("name")["selection_score"].mean().idxmax()
        method, k = best_angle.rsplit("-k", 1)
        cfg = replace(cfg, angle_method=method, k_angle=int(k))
    res = run_protocol("D", cfg, index, feats, DATA_SHA, run_name="export-v1.0.0")
    model = res["model"]
    model.paths = np.array([Path(p).name for p in model.paths], dtype=object)
    model.save(out)

    from .quality import quality_refs
    split_rows = np.flatnonzero(index.obj_id.isin(set(model.obj_ids)).to_numpy())
    thresholds = {**res["thresholds"], "quality": asdict(quality_refs(images[split_rows]))}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
    manifest = {
        "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": DATA_SHA,
        "train_split": "D (90 objetos conocidos, vistas cada 10°)",
        "unknown_objs": sorted(set(range(1, 101)) - set(int(o) for o in model.obj_ids)),
        "config": asdict(cfg),
        "model_sha256": hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Exportado", out, f"{(out / 'model.joblib').stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "angle", "export"], required=True)
    args = ap.parse_args()
    index, images, feats = load_all()
    {"grid": stage_grid, "angle": stage_angle, "export": stage_export}[args.stage](index, images, feats)


if __name__ == "__main__":
    main()
