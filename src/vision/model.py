"""Modelo de visión clásico: clase, ángulo, vistas similares y señales de confianza.

La clase y el ángulo usan espacios de características distintos a propósito:
para identificar conviene una representación que no cambie al girar el objeto,
para el ángulo una que sí cambie.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .circular import circ_mean, circ_std
from .features import build_features
from .symmetry import rotation_distances, symmetry_period


@dataclass
class Neighbor:
    obj_id: int
    angle: int
    distance: float
    path: str


@dataclass
class Prediction:
    obj_id: int | None
    class_prob: float
    ood_distance: float
    angle: float | None
    angle_std: float
    symmetry: int
    topk: list[Neighbor] = field(default_factory=list)


@dataclass
class ModelConfig:
    classifier: Literal["knn", "svm", "rf", "logreg"] = "svm"
    class_features: list[str] = field(default_factory=lambda: ["color", "hog"])
    angle_features: list[str] = field(default_factory=lambda: ["hog", "pixels"])
    pca_dims: int = 128
    k_angle: int = 2
    angle_method: Literal["knn", "kernel_ridge"] = "knn"
    calibrate: bool = True
    seed: int = 42


def _classifier(name: str, seed: int):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(name)


def _projector(n_features: int, n_samples: int, dims: int, seed: int) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if dims and dims < min(n_features, n_samples):
        steps.append(("pca", PCA(n_components=dims, random_state=seed)))
    return Pipeline(steps)


def featurize(images: np.ndarray | None, kinds: list[str], precomputed: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Concatena características; las de `precomputed` (p. ej. "cnn") no se recalculan."""
    precomputed = precomputed or {}
    parts = [precomputed[k] if k in precomputed else build_features(images, [k]) for k in kinds]
    return np.concatenate(parts, axis=1).astype(np.float32)


class VisionModel:
    def __init__(self, config: ModelConfig):
        self.config = config

    # ---------- entrenamiento ----------
    def fit(self, images, obj_ids, angles, paths, config: ModelConfig | None = None,
            precomputed: dict[str, np.ndarray] | None = None) -> "VisionModel":
        cfg = self.config = config or self.config
        self.obj_ids = np.asarray(obj_ids, dtype=int)
        self.angles = np.asarray(angles, dtype=int)
        self.paths = np.asarray(paths, dtype=object)

        Xc = featurize(images, cfg.class_features, precomputed)
        self.class_proj = _projector(Xc.shape[1], len(Xc), cfg.pca_dims, cfg.seed).fit(Xc)
        Zc = self.class_proj.transform(Xc)
        base = _classifier(cfg.classifier, cfg.seed)
        if cfg.classifier == "svm":
            base.set_params(probability=True)
        self.calibrator = None
        min_per_class = np.bincount(self.obj_ids)[np.unique(self.obj_ids)].min()
        if cfg.calibrate and min_per_class >= 3:
            # Calibración de la etiqueta ganadora: la clase la decide el clasificador sin tocar y
            # su confianza se ajusta con isotónica sobre predicciones fuera de fold.
            # Folds barajados: las filas vienen ordenadas por ángulo y sin barajar cada fold
            # perdería un sector entero de vistas.
            folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.seed)
            oof = cross_val_predict(clone(base), Zc, self.obj_ids, cv=folds, method="predict_proba")
            classes = np.unique(self.obj_ids)
            top = oof.max(axis=1)
            correct = classes[oof.argmax(axis=1)] == self.obj_ids
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(top, correct)
        self.clf = base.fit(Zc, self.obj_ids)
        self.class_nn = NearestNeighbors(n_neighbors=10).fit(Zc)

        Xa = featurize(images, cfg.angle_features, precomputed)
        self.angle_proj = _projector(Xa.shape[1], len(Xa), cfg.pca_dims, cfg.seed).fit(Xa)
        Za = self.angle_proj.transform(Xa)
        self.angle_index: dict[int, tuple[NearestNeighbors, np.ndarray, np.ndarray]] = {}
        self.angle_reg: dict[int, KernelRidge] = {}
        self.symmetry: dict[int, int] = {}
        d90s = [rotation_distances(Za[self.obj_ids == o], self.angles[self.obj_ids == o])[0]
                for o in np.unique(self.obj_ids)]
        self.ref_d90 = float(np.median([d for d in d90s if d is not None]))
        for o in np.unique(self.obj_ids):
            rows = np.flatnonzero(self.obj_ids == o)
            Z, ang = Za[rows], self.angles[rows]
            self.symmetry[int(o)] = symmetry_period(Z, ang, self.ref_d90)
            k = min(cfg.k_angle, len(rows))
            self.angle_index[int(o)] = (NearestNeighbors(n_neighbors=k).fit(Z), ang, rows)
            if cfg.angle_method == "kernel_ridge":
                target = self._angle_target(ang, self.symmetry[int(o)])
                self.angle_reg[int(o)] = KernelRidge(kernel="rbf", alpha=0.1, gamma=1.0 / Z.shape[1]).fit(Z, target)
        return self

    @staticmethod
    def _angle_target(ang: np.ndarray, sym: int) -> np.ndarray:
        mult = 2 if sym == 180 else 1
        r = np.radians(ang * mult)
        return np.column_stack([np.sin(r), np.cos(r)])

    # ---------- inferencia ----------
    def predict(self, images, top_k: int = 5, precomputed: dict[str, np.ndarray] | None = None) -> list[Prediction]:
        cfg = self.config
        Zc = self.class_proj.transform(featurize(images, cfg.class_features, precomputed))
        Za = self.angle_proj.transform(featurize(images, cfg.angle_features, precomputed))
        proba = self.clf.predict_proba(Zc)
        classes = self.clf.classes_
        dist, nbr = self.class_nn.kneighbors(Zc, n_neighbors=max(top_k, 1))

        preds = []
        for i in range(len(Zc)):
            j = int(np.argmax(proba[i]))
            obj = int(classes[j])
            angle, std = self._estimate_angle(obj, Za[i:i + 1])
            topk = [Neighbor(int(self.obj_ids[n]), int(self.angles[n]), float(d), str(self.paths[n]))
                    for d, n in zip(dist[i][:top_k], nbr[i][:top_k])]
            prob = float(proba[i, j])
            if self.calibrator is not None:
                prob = float(self.calibrator.predict([prob])[0])
            preds.append(Prediction(obj, prob, float(dist[i, 0]), angle, std, self.symmetry[obj], topk))
        return preds

    def _estimate_angle(self, obj: int, z: np.ndarray) -> tuple[float | None, float]:
        sym = self.symmetry[obj]
        if sym == 0:
            return 0.0, 0.0
        mult = 2 if sym == 180 else 1
        if self.config.angle_method == "kernel_ridge" and obj in self.angle_reg:
            s, c = self.angle_reg[obj].predict(z)[0]
            R = min(max(np.hypot(s, c), 1e-6), 1.0)
            ang = np.degrees(np.arctan2(s, c)) % 360 / mult
            return float(ang), float(np.degrees(np.sqrt(-2 * np.log(R))) / mult)
        nn, ang, _ = self.angle_index[obj]
        d, idx = nn.kneighbors(z)
        w = 1.0 / (d[0] + 1e-6)
        a = ang[idx[0]] * mult
        return float(circ_mean(a, w) / mult), float(circ_std(a, w) / mult)

    # ---------- persistencia ----------
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_dir / "model.joblib", compress=3)

    @classmethod
    def load(cls, model_dir: Path) -> "VisionModel":
        # joblib deserializa con pickle: solo se carga el artefacto propio del repo y,
        # si hay manifiesto, después de comprobar su SHA-256.
        path = Path(model_dir) / "model.joblib"
        manifest = Path(model_dir) / "manifest.json"
        if manifest.exists():
            expected = json.loads(manifest.read_text(encoding="utf-8")).get("model_sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("El SHA-256 de model.joblib no coincide con el manifiesto")
        return joblib.load(path)


def prediction_to_dict(p: Prediction) -> dict:
    return asdict(p)
