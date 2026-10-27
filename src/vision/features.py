"""Extractores de características clásicos.

color: histograma HSV, casi no cambia al girar el objeto (sirve para la clase).
hog, pixels: cambian con la orientación (sirven para el ángulo).
lbp: textura local.
"""
import hashlib
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

Kind = Literal["color", "hog", "lbp", "pixels"]


def _color(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = (img.max(axis=2) > 20).astype(np.uint8)  # ignora el fondo negro
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def _gray(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size, size), interpolation=cv2.INTER_AREA)


def _hog(img: np.ndarray) -> np.ndarray:
    return hog(_gray(img, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


def _lbp(img: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(_gray(img, 128), P=8, R=1, method="uniform")
    feats = []
    for rows in np.array_split(lbp, 4, axis=0):
        for cell in np.array_split(rows, 4, axis=1):
            h, _ = np.histogram(cell, bins=10, range=(0, 10))
            feats.append(h / max(h.sum(), 1))
    return np.concatenate(feats)


def _pixels(img: np.ndarray) -> np.ndarray:
    return _gray(img, 32).ravel() / 255.0


_FN = {"color": _color, "hog": _hog, "lbp": _lbp, "pixels": _pixels}


def extract(images: np.ndarray, kind: Kind) -> np.ndarray:
    fn = _FN[kind]
    return np.stack([fn(img) for img in images]).astype(np.float32)


def build_features(images: np.ndarray, kinds: list[str]) -> np.ndarray:
    return np.concatenate([extract(images, k) for k in kinds], axis=1)


def cache_features(images: np.ndarray, kinds: list[str], cache_path: Path, data_sha: str) -> np.ndarray:
    """Calcula o reutiliza las características guardadas para estas imágenes y tipos."""
    key = hashlib.sha256(f"{data_sha}|{','.join(kinds)}|{len(images)}".encode()).hexdigest()
    cache_path = Path(cache_path)
    if cache_path.exists():
        d = np.load(cache_path)
        if str(d["key"]) == key:
            return d["X"]
    X = build_features(images, kinds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, key=key)
    return X
