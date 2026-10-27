from pathlib import Path

import numpy as np
import pytest

from src.vision.circular import circ_dist
from src.vision.data import load_images, load_index
from src.vision.model import ModelConfig, Prediction, VisionModel

RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="ejecutar antes data/download.py")


@pytest.fixture(scope="module")
def small():
    index = load_index(RAW)
    index = index[index.obj_id.isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    images = load_images(index)
    train = (index.angle % 10 == 0).to_numpy()
    return index, images, train


@pytest.fixture(scope="module", params=["svm", "knn"])
def fitted(request, small):
    index, images, train = small
    cfg = ModelConfig(classifier=request.param, pca_dims=32)
    m = VisionModel(cfg).fit(images[train], index.obj_id[train], index.angle[train], index.path[train])
    return m


def test_predict_types_and_accuracy(fitted, small):
    index, images, train = small
    preds = fitted.predict(images[~train])
    assert all(isinstance(p, Prediction) for p in preds)
    assert {p.obj_id for p in preds} <= {1, 2, 3, 4, 5}
    acc = np.mean([p.obj_id == t for p, t in zip(preds, index.obj_id[~train])])
    assert acc > 0.9


def test_training_view_angle(fitted, small):
    index, images, train = small
    rows = np.flatnonzero(train)[:20]
    preds = fitted.predict(images[rows])
    for p, r in zip(preds, rows):
        if p.symmetry == 360 and p.obj_id == index.obj_id[r]:
            assert circ_dist(p.angle, index.angle[r]) <= 5


def test_topk_sorted(fitted, small):
    _, images, _ = small
    p = fitted.predict(images[:1], top_k=5)[0]
    d = [n.distance for n in p.topk]
    assert len(d) == 5 and d == sorted(d)


def test_save_load_identical(fitted, small, tmp_path):
    _, images, _ = small
    fitted.save(tmp_path)
    loaded = VisionModel.load(tmp_path)
    a, b = fitted.predict(images[:5]), loaded.predict(images[:5])
    assert [(p.obj_id, p.angle) for p in a] == [(p.obj_id, p.angle) for p in b]


def test_kernel_ridge_angle(small):
    index, images, train = small
    cfg = ModelConfig(classifier="logreg", pca_dims=32, angle_method="kernel_ridge")
    m = VisionModel(cfg).fit(images[train], index.obj_id[train], index.angle[train], index.path[train])
    p = m.predict(images[~train][:3])
    assert all(p_.angle is not None and 0 <= p_.angle < 360 for p_ in p)
