import numpy as np
import pytest

from src.vision.metrics import auroc, choose_threshold, ece, pick_outcome, risk_coverage, within_tol


def test_within_tol():
    assert within_tol(np.array([0, 5, 10, 11])) == 0.75


def test_ece_perfect_calibration():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1, 1, 0, 0])
    assert ece(conf, correct) == pytest.approx(0.0)


def test_ece_overconfident():
    conf = np.full(10, 0.9)
    correct = np.array([1] * 5 + [0] * 5)
    assert ece(conf, correct) == pytest.approx(0.4)


def test_risk_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    cov, risk = risk_coverage(conf, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(np.round(risk, 4)) == [0.0, 0.0, 0.3333, 0.25]


def test_choose_threshold():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    assert choose_threshold(conf, correct, max_risk=0.01) == 0.8
    assert choose_threshold(conf, correct, max_risk=0.30) == 0.6
    assert choose_threshold(np.array([0.9]), np.array([0]), max_risk=0.01) == np.inf


def test_auroc():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5


def test_pick_outcome():
    out = pick_outcome(pred_obj=np.array([1, 2, 3, 4]), true_obj=np.array([1, 2, 9, 4]),
                       err_deg=np.array([3, 20, 1, 5]), accepted=np.array([1, 1, 1, 0], dtype=bool))
    assert out["coverage"] == 0.75
    assert out["wrong_pick_rate"] == pytest.approx(2 / 3)  # clase errónea y ángulo de 20°
    assert out["class_acc"] == 0.75
    assert out["angle_within_tol"] == pytest.approx(2 / 3)  # entre los de clase correcta


def test_choose_threshold_weighted():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    # El error a 0,7 pesa poco: con peso 0,01 el riesgo ponderado a 0,6 es 0,01/3,01 < 1 %
    assert choose_threshold(conf, correct, 0.01, weights=np.array([1, 1, 0.01, 1])) == 0.6
