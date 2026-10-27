import json
from pathlib import Path

import numpy as np
import pytest

from src.vision.data import load_images, load_index
from src.vision.model import ModelConfig, VisionModel

RAW = Path("data/raw")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Modelo pequeño (5 objetos) con umbrales y manifiesto, para probar la API sin el artefacto final."""
    if not RAW.exists():
        pytest.skip("sin datos")
    index = load_index(RAW)
    index = index[index.obj_id.isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    tr = (index.angle % 10 == 0).to_numpy()
    images = load_images(index)
    m = VisionModel(ModelConfig(classifier="svm", pca_dims=32)).fit(
        images[tr], index.obj_id[tr], index.angle[tr], [Path(p).name for p in index.path[tr]])
    d = tmp_path_factory.mktemp("model")
    m.save(d)
    th = {"max_ood": 1e9, "min_pick_score": 0.5,
          "pick_model": {"coef": [10.0, 0.0, -1.0, 0.0, 0.0], "intercept": -5.0, "ood_scale": 1.0},
          "quality": {"blur_min": 1.0, "brightness_min": 1.0, "brightness_max": 255.0, "fg_ratio_min": 0.01}}
    (d / "thresholds.json").write_text(json.dumps(th))
    (d / "manifest.json").write_text(json.dumps({"version": "test", "unknown_objs": []}))
    return d
