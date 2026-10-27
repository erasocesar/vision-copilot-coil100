import numpy as np
import pytest

from src.vision.features import build_features, cache_features, extract

DIMS = {"color": 128, "hog": 1764, "lbp": 160, "pixels": 1024}


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    imgs = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    for i in range(4):
        # objeto asimétrico sobre fondo negro
        imgs[i, 30:100, 20 + 5 * i:70 + 5 * i] = rng.integers(50, 255, size=3)
        imgs[i, 40:60, 80:110] = [200, 30, 30]
    return imgs


@pytest.mark.parametrize("kind", list(DIMS))
def test_shape_and_dtype(images, kind):
    f = extract(images, kind)
    assert f.shape == (4, DIMS[kind]) and f.dtype == np.float32


def test_deterministic(images):
    assert np.array_equal(build_features(images, ["color", "hog"]), build_features(images, ["color", "hog"]))


def test_concat(images):
    assert build_features(images, ["color", "lbp"]).shape == (4, 128 + 160)


def test_color_invariant_to_flip_hog_not(images):
    flipped = images[:, :, ::-1]
    c = np.abs(extract(images, "color") - extract(flipped, "color")).sum(axis=1)
    h = np.abs(extract(images, "hog") - extract(flipped, "hog")).sum(axis=1)
    assert (c < 0.05).all()
    assert (h > 0.5).all()


def test_cache(images, tmp_path):
    p = tmp_path / "f.npz"
    a = cache_features(images, ["color"], p, data_sha="x")
    b = cache_features(images, ["color"], p, data_sha="x")
    assert p.exists() and np.array_equal(a, b)
