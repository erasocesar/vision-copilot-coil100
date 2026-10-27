import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.vision.model import Prediction
from src.vision.policy import Decision, Thresholds, decide
from src.vision.quality import InvalidImage, QualityRefs, QualityReport, check_quality, decode_image, quality_refs

SAMPLE = Path("data/raw/obj1__0.png")
REFS = QualityRefs(blur_min=50.0, brightness_min=5.0, brightness_max=200.0, fg_ratio_min=0.02)
PICK_MODEL = {"coef": [10.0, -2.0, -1.0, 0.0, 0.0], "intercept": -5.0, "ood_scale": 30.0}
TH = Thresholds(max_ood=30.0, min_pick_score=0.9, pick_model=PICK_MODEL)
OK_Q = QualityReport(ok=True, blur=500, brightness=60, fg_ratio=0.3, reasons=[])


def _png(arr: np.ndarray, mode=None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _pred(**kw):
    base = dict(obj_id=1, class_prob=0.99, ood_distance=10.0, angle=45.0, angle_std=1.0, symmetry=360, topk=[])
    base.update(kw)
    return Prediction(**base)


def test_decode_rejects_non_image():
    with pytest.raises(InvalidImage):
        decode_image(b"esto no es una imagen")


def test_decode_rejects_too_big():
    with pytest.raises(InvalidImage):
        decode_image(b"\x89PNG" + b"0" * 6_000_000)


def test_decode_gray_small_to_rgb_128():
    img = decode_image(_png(np.full((64, 64), 120, dtype=np.uint8)))
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8


def test_decode_rgba():
    img = decode_image(_png(np.full((128, 128, 4), 120, dtype=np.uint8)))
    assert img.shape == (128, 128, 3)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sin datos")
def test_blurred_image_fails_quality():
    img = decode_image(SAMPLE.read_bytes())
    assert check_quality(img, REFS).ok
    blurred = cv2.GaussianBlur(img, (0, 0), 6)
    q = check_quality(blurred, REFS)
    assert not q.ok and "imagen desenfocada" in q.reasons


def test_black_image_no_object():
    q = check_quality(np.zeros((128, 128, 3), dtype=np.uint8), REFS)
    assert not q.ok and "no se ve un objeto aislado" in q.reasons


def test_quality_refs_from_images():
    imgs = np.zeros((3, 128, 128, 3), dtype=np.uint8)
    imgs[:, 40:90, 40:90] = 200
    r = quality_refs(imgs)
    assert r.fg_ratio_min > 0 and r.brightness_max >= r.brightness_min


def test_decide_pick():
    d = decide(_pred(), OK_Q, TH)
    assert isinstance(d, Decision) and d.action == "PICK" and d.angle == 45.0 and d.angle_mode == "exact"
    assert 0.9 <= d.pick_score <= 1.0


def test_decide_divert_quality():
    q = QualityReport(ok=False, blur=1, brightness=60, fg_ratio=0.3, reasons=["imagen desenfocada"])
    d = decide(_pred(), q, TH)
    assert d.action == "DIVERT" and "imagen desenfocada" in d.reasons


def test_decide_divert_unknown():
    d = decide(_pred(ood_distance=50.0), OK_Q, TH)
    assert d.action == "DIVERT" and any("desconocido" in r for r in d.reasons)


def test_decide_divert_low_class_confidence():
    d = decide(_pred(class_prob=0.40), OK_Q, TH)
    assert d.action == "DIVERT" and any("clase" in r for r in d.reasons)


def test_decide_divert_angle_uncertain():
    d = decide(_pred(angle_std=40.0), OK_Q, TH)
    assert d.action == "DIVERT" and any("ángulo" in r for r in d.reasons)


def test_decide_full_symmetry_any_angle():
    d = decide(_pred(symmetry=0, angle=0.0, angle_std=0.0), OK_Q, TH)
    assert d.action == "PICK" and d.angle_mode == "any"


def test_decide_180_symmetry_mod180():
    d = decide(_pred(symmetry=180, angle=200.0), OK_Q, TH)
    assert d.angle_mode == "mod180" and d.angle == 20.0


def test_decode_rejects_other_formats():
    buf = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8)).save(buf, format="BMP")
    with pytest.raises(InvalidImage):
        decode_image(buf.getvalue())
