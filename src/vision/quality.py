"""Lectura segura de imágenes y control de calidad antes de inferir."""
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 5_000_000
MAX_SIDE = 4096
SIZE = 128
FG_LEVEL = 20  # en COIL el fondo es negro: un píxel con algún canal > 20 es objeto


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes, max_bytes: int = MAX_BYTES) -> np.ndarray:
    """Bytes de PNG/JPEG a RGB uint8 128x128. Lanza InvalidImage si no se puede usar."""
    if len(data) > max_bytes:
        raise InvalidImage(f"la imagen pesa más de {max_bytes // 1_000_000} MB")
    try:
        # Solo PNG y JPEG: otros decodificadores de Pillow (EPS, por ejemplo) llaman a programas externos.
        img = Image.open(io.BytesIO(data), formats=["PNG", "JPEG"])
        if img.format not in ("PNG", "JPEG"):
            raise InvalidImage("formato no admitido (solo PNG o JPEG)")
        if max(img.size) > MAX_SIDE:
            raise InvalidImage(f"la imagen mide más de {MAX_SIDE} px por lado")
        arr = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as e:
        if isinstance(e, InvalidImage):
            raise
        raise InvalidImage("el archivo no es una imagen válida") from None
    if arr.shape[:2] != (SIZE, SIZE):
        arr = cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr, dtype=np.uint8)


@dataclass
class QualityRefs:
    blur_min: float
    brightness_min: float
    brightness_max: float
    fg_ratio_min: float


@dataclass
class QualityReport:
    ok: bool
    blur: float
    brightness: float
    fg_ratio: float
    reasons: list[str] = field(default_factory=list)


def _measures(img: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fg = img.max(axis=2) > FG_LEVEL
    brightness = float(gray[fg].mean()) if fg.any() else 0.0
    return blur, brightness, float(fg.mean())


def quality_refs(images: np.ndarray) -> QualityRefs:
    """Referencias de calidad: percentiles 1 y 99 de las imágenes de entrenamiento."""
    m = np.array([_measures(img) for img in images])
    return QualityRefs(blur_min=float(np.percentile(m[:, 0], 1)),
                       brightness_min=float(np.percentile(m[:, 1], 1)),
                       brightness_max=float(np.percentile(m[:, 1], 99)),
                       fg_ratio_min=float(np.percentile(m[:, 2], 1)))


def check_quality(img: np.ndarray, refs: QualityRefs) -> QualityReport:
    blur, brightness, fg_ratio = _measures(img)
    reasons = []
    if fg_ratio < refs.fg_ratio_min:
        reasons.append("no se ve un objeto aislado")
    else:
        if blur < refs.blur_min:
            reasons.append("imagen desenfocada")
        if brightness < refs.brightness_min:
            reasons.append("imagen subexpuesta")
        if brightness > refs.brightness_max:
            reasons.append("imagen sobreexpuesta")
    return QualityReport(not reasons, blur, brightness, fg_ratio, reasons)
