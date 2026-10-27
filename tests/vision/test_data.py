from pathlib import Path

import numpy as np
import pytest

from src.vision.data import load_images, load_index

RAW = Path("data/raw")

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="ejecutar antes data/download.py")


def test_index_shape():
    index = load_index(RAW)
    assert len(index) == 7200
    assert index["obj_id"].nunique() == 100
    assert set(index["obj_id"]) == set(range(1, 101))
    per_obj = index.groupby("obj_id")["angle"].nunique()
    assert (per_obj == 72).all()
    assert (index["angle"] % 5 == 0).all()
    assert index["angle"].min() == 0 and index["angle"].max() == 355
    assert list(index[["obj_id", "angle"]].iloc[:2].itertuples(index=False, name=None)) == [(1, 0), (1, 5)]


def test_images_dtype():
    index = load_index(RAW)
    images = load_images(index.head(3))
    assert images.shape == (3, 128, 128, 3)
    assert images.dtype == np.uint8
