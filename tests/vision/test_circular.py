import numpy as np
import pytest

from src.vision.circular import circ_dist, circ_mean, circ_std, sym_dist
from src.vision.symmetry import symmetry_period


def test_circ_dist():
    assert circ_dist(355, 5) == 10
    assert circ_dist(0, 180) == 180
    assert circ_dist(90, 90) == 0
    assert list(circ_dist(np.array([10, 350]), np.array([350, 10]))) == [20, 20]


def test_circ_mean_wraps_zero():
    m = circ_mean(np.array([355, 5]))
    assert min(m, 360 - m) == pytest.approx(0, abs=1e-6)
    assert 0 <= m < 360


def test_circ_mean_weighted():
    m = circ_mean(np.array([350, 10]), weights=np.array([3, 1]))
    assert circ_dist(m, 355) < 1.0


def test_circ_std():
    assert circ_std(np.array([10, 10, 10])) == pytest.approx(0, abs=1e-6)
    assert circ_std(np.array([0, 10])) > circ_std(np.array([0, 2]))
    assert circ_std(np.array([0, 10])) == pytest.approx(5.0, abs=0.1)


def test_sym_dist():
    assert sym_dist(10, 190, 180) == 0
    assert sym_dist(10, 100, 180) == 90
    assert sym_dist(10, 200, 0) == 0
    assert sym_dist(10, 200, 360) == 170


def _views(fn):
    angles = np.arange(0, 360, 10)
    return np.stack([fn(a) for a in angles]), angles


def test_symmetry_full():
    feats, angles = _views(lambda a: np.ones(16))
    assert symmetry_period(feats, angles) == 0


def test_symmetry_180():
    # vistas que cambian de forma continua y se repiten cada 180°
    feats, angles = _views(lambda a: np.array([np.cos(np.radians(2 * a)), np.sin(np.radians(2 * a))] * 8))
    assert symmetry_period(feats, angles) == 180


def test_symmetry_none():
    feats, angles = _views(lambda a: np.array([np.cos(np.radians(a)), np.sin(np.radians(a))] * 8))
    assert symmetry_period(feats, angles) == 360
