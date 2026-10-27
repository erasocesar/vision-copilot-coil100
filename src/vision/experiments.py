"""Ejecución de un protocolo: umbrales en validación, una sola evaluación en prueba, registro en MLflow."""
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from .circular import circ_dist, sym_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .metrics import auroc, choose_threshold, ece, pick_outcome, within_tol
from .model import ModelConfig, VisionModel
from .policy import pick_features, pick_probability
from .protocols import Split, make_split, make_validation

MAX_PICK_RISK = 0.01
OOD_KNOWN_QUANTILE = 0.99


def fit_predict(cfg: ModelConfig, index: pd.DataFrame, feats: dict[str, np.ndarray],
                train_rows: np.ndarray, test_rows: np.ndarray) -> tuple[VisionModel, pd.DataFrame, float]:
    tr = index.iloc[train_rows]
    model = VisionModel(cfg).fit(None, tr.obj_id, tr.angle, tr.path,
                                 precomputed={k: v[train_rows] for k, v in feats.items()})
    t0 = time.perf_counter()
    preds = model.predict(None, top_k=3, precomputed={k: v[test_rows] for k, v in feats.items()})
    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(test_rows), 1)
    return model, predictions_frame(preds, index, test_rows, set(tr.obj_id.unique())), latency_ms


def predictions_frame(preds, index: pd.DataFrame, rows: np.ndarray, known_objs: set[int]) -> pd.DataFrame:
    te = index.iloc[rows]
    df = pd.DataFrame({
        "row": rows,
        "true_obj": te.obj_id.to_numpy(),
        "true_angle": te.angle.to_numpy(),
        "pred_obj": [p.obj_id for p in preds],
        "prob": [p.class_prob for p in preds],
        "ood": [p.ood_distance for p in preds],
        "angle": [p.angle for p in preds],
        "angle_std": [p.angle_std for p in preds],
        "sym": [p.symmetry for p in preds],
    })
    df["known"] = df.true_obj.isin(known_objs)
    df["cls_ok"] = df.known & (df.pred_obj == df.true_obj)
    df["err"] = circ_dist(df.angle.astype(float), df.true_angle)
    df["err_sym"] = [float(sym_dist(a, t, s)) for a, t, s in zip(df.angle, df.true_angle, df.sym)]
    return df


def pick_score(df: pd.DataFrame, th: dict) -> np.ndarray:
    return pick_probability(df.prob, df.ood, df.angle_std, df.sym, th["pick_model"])


def choose_thresholds(val: pd.DataFrame, max_risk: float = MAX_PICK_RISK) -> dict:
    """Política elegida solo con validación.

    1) Puerta de desconocidos: distancia al catálogo por encima del percentil 99 de los conocidos.
    2) Puntuación de agarre: regresión logística sobre (confianza de clase, distancia, incertidumbre
       del ángulo, simetría) que estima la probabilidad de agarre correcto (clase bien y ángulo <= 10°).
    3) Umbral de la puntuación: máxima cobertura con riesgo <= max_risk, medido con predicciones
       fuera de fold para no evaluar la regresión con los mismos datos que la ajustan.
    """
    known = val[val.known]
    max_ood = float(np.quantile(known.ood, OOD_KNOWN_QUANTILE))
    good = (val.cls_ok & (val.err_sym <= 10)).to_numpy()
    X = pick_features(val.prob, val.ood, val.angle_std, val.sym, max_ood)
    lr = LogisticRegression(max_iter=1000)
    if good.all() or not good.any():
        score_oof = np.full(len(val), float(good.mean()))
        lr = None
    else:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        score_oof = cross_val_predict(lr, X, good, cv=folds, method="predict_proba")[:, 1]
        lr.fit(X, good)
    gate = (val.ood <= max_ood).to_numpy()
    t = choose_threshold(score_oof[gate], good[gate], max_risk) if gate.any() else np.inf
    return {
        "max_ood": max_ood,
        "min_pick_score": float(t),
        "max_risk": max_risk,
        "pick_model": {"coef": (lr.coef_[0].tolist() if lr else [0.0] * X.shape[1]),
                       "intercept": float(lr.intercept_[0]) if lr else float(np.log(good.mean() + 1e-9)),
                       "ood_scale": max_ood,
                       "features": ["class_prob", "ood/ood_scale", "angle_std/10 (0 si simétrico)",
                                    "simétrico total", "simétrico 180"]},
    }


def accepted_mask(df: pd.DataFrame, th: dict) -> np.ndarray:
    return ((df.ood <= th["max_ood"]).to_numpy() & (pick_score(df, th) >= th["min_pick_score"]))


def summarize(df: pd.DataFrame, th: dict, prefix: str) -> dict:
    acc = accepted_mask(df, th)
    known = df[df.known]
    out = pick_outcome(df.pred_obj.where(df.known, -1), df.true_obj, df.err_sym, acc)
    m = {
        "class_acc": float(known.cls_ok.mean()),
        "angle_within_tol": within_tol(known[known.cls_ok].err),
        "angle_within_tol_sym": within_tol(known[known.cls_ok].err_sym),
        "angle_median_err_sym": float(known[known.cls_ok].err_sym.median()),
        "ece": ece(known.prob, known.cls_ok),
        "coverage": out["coverage"],
        "wrong_pick_rate": out["wrong_pick_rate"],
    }
    if (~df.known).any():
        unk = df[~df.known]
        m["unknown_accept_rate"] = float(acc[~df.known.to_numpy()].mean())
        m["auroc_ood"] = auroc(-known.ood, -unk.ood)
        m["auroc_prob"] = auroc(known.prob, unk.prob)
    return {f"{prefix}_{k}": v for k, v in m.items()}


def validation_frame(cfg, index, feats, split: Split) -> pd.DataFrame:
    vals = make_validation(index, split)
    vals = vals if isinstance(vals, list) else [vals]
    return pd.concat([fit_predict(cfg, index, feats, v.train_idx, v.test_idx)[1] for v in vals], ignore_index=True)


def run_protocol(protocol: str, cfg: ModelConfig, index: pd.DataFrame, feats: dict[str, np.ndarray],
                 data_sha: str, mlflow_uri: str | None = "sqlite:///mlruns/mlflow.db", run_name: str = "") -> dict:
    split = make_split(index, protocol)
    val = validation_frame(cfg, index, feats, split)
    th = choose_thresholds(val)
    model, test, latency = fit_predict(cfg, index, feats, split.train_idx, split.test_idx)
    metrics = {**summarize(val, th, "val"), **summarize(test, th, "test"), "latency_ms": latency}
    for r in (0.02, 0.05):
        th_r = choose_thresholds(val, r)
        acc = accepted_mask(test, th_r)
        good = (test.cls_ok & (test.err_sym <= 10)).to_numpy()
        metrics[f"test_coverage_r{int(r * 100)}"] = float(acc.mean())
        metrics[f"test_wrong_pick_rate_r{int(r * 100)}"] = float((~good[acc]).mean()) if acc.any() else 0.0

    if mlflow_uri:
        import mlflow
        Path("mlruns/artifacts").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(mlflow_uri)
        exp = f"protocol_{protocol}"
        if mlflow.get_experiment_by_name(exp) is None:
            mlflow.create_experiment(exp, artifact_location=(Path("mlruns/artifacts") / exp).resolve().as_uri())
        mlflow.set_experiment(exp)
        with mlflow.start_run(run_name=run_name or f"{cfg.classifier}-{'+'.join(cfg.class_features)}"):
            params = {k: (",".join(v) if isinstance(v, list) else v) for k, v in asdict(cfg).items()}
            mlflow.log_params({**params, "protocol": protocol, "data_sha256": data_sha})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            mlflow.log_dict(th, "thresholds.json")
            mlflow.log_text(test.to_csv(index=False), "test_predictions.csv")
    return {"protocol": protocol, "config": cfg, "thresholds": th, "metrics": metrics,
            "model": model, "val": val, "test": test}
