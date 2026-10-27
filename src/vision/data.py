"""Índice e imágenes de COIL-100."""
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FILE_RE = re.compile(r"^obj(\d{1,3})__(\d{1,3})\.png$")


def load_index(raw_dir: Path) -> pd.DataFrame:
    """Una fila por imagen: obj_id (1..100), angle (0..355), path."""
    rows = []
    for p in Path(raw_dir).iterdir():
        m = FILE_RE.match(p.name)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), str(p)))
    index = pd.DataFrame(rows, columns=["obj_id", "angle", "path"])
    return index.sort_values(["obj_id", "angle"]).reset_index(drop=True)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(index: pd.DataFrame) -> np.ndarray:
    """Imágenes RGB uint8 con forma (N, 128, 128, 3), en el orden del índice."""
    return np.stack([read_rgb(p) for p in index["path"]])


def write_manifest(index: pd.DataFrame, out: Path) -> None:
    """Manifiesto con el SHA-256 de cada imagen (versión de datos)."""
    m = index.copy()
    m["sha256"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in m["path"]]
    m["path"] = [Path(p).name for p in m["path"]]
    m.to_csv(out, index=False)
