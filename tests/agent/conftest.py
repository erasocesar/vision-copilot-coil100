from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.agent.context import ToolContext
from src.agent.vision_client import VisionError
from src.common.db import connect, init_schema
from src.sim.line_sim import seed_state, simulate
from tests.sim.test_line_sim import fake_predict

IMG_ID = "a" * 64


class FakeVision:
    def __init__(self):
        self.fault = None
        self.calls = []

    def _check(self):
        if self.fault == "500":
            raise VisionError("server", "HTTP 503")
        if self.fault == "timeout":
            raise VisionError("timeout", "lento")

    def quality(self, path):
        self._check()
        self.calls.append(("quality", str(path)))
        return {"image_id": IMG_ID, "quality": {"ok": True, "blur": 900.0, "brightness": 80.0, "fg_ratio": 0.3,
                                                "reasons": []}}

    def predict(self, path, top_k=3):
        self._check()
        self.calls.append(("predict", str(path)))
        return {"image_id": IMG_ID,
                "quality": {"ok": True, "blur": 900.0, "brightness": 80.0, "fg_ratio": 0.3, "reasons": []},
                "prediction": {"obj_id": 7, "class_prob": 0.99, "ood_distance": 10.0, "angle": 120.0,
                               "angle_std": 2.0, "symmetry": 360,
                               "topk": [{"obj_id": 7, "angle": 120, "distance": 3.0, "path": "obj7__120.png"}] * top_k},
                "decision": {"action": "PICK", "obj_id": 7, "angle": 120.0, "angle_mode": "exact",
                             "pick_score": 0.995, "reasons": []},
                "model_version": "1.0.0", "latency_ms": 5.0}

    def evaluate_views(self, paths, angles):
        self._check()
        return {"n_views": len(angles), "max_gap_deg": 10, "symmetry": 360, "loo_within_tol": 0.9,
                "meets_criteria": True, "reasons": []}


@pytest.fixture
def db():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(20, 128, 128, 3), dtype=np.uint8)
    index = pd.DataFrame({"obj_id": [1] * 20, "angle": [10] * 20})
    c = connect(":memory:")
    init_schema(c)
    seed_state(c, "P1", "L1", 0.97, "1.0.0")
    seed_state(c, "P2", "L1", 0.97, "1.0.0")
    simulate(c, fake_predict, images, index, datetime(2026, 10, 26, 6), "P1", "L1")
    simulate(c, fake_predict, images, index, datetime(2026, 10, 26, 6), "P2", "L1", rate_per_min=0.1)
    return c


@pytest.fixture
def image_dir(tmp_path):
    (tmp_path / f"{IMG_ID}.png").write_bytes(b"png")
    for i in range(12):
        (tmp_path / f"{i:064x}.png").write_bytes(b"png")
    return tmp_path


@pytest.fixture
def make_ctx(db, image_dir):
    vision = FakeVision()

    def _make(role="supervisor", user="ana", plant="P1"):
        return ToolContext(user, role, plant, "L1", "s1", db, vision, str(image_dir))
    _make.vision = vision
    return _make
