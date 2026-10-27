from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common.db import connect, init_schema, rows
from src.sim.line_sim import seed_state, simulate
from src.vision.model import Prediction
from src.vision.policy import Decision
from src.vision.quality import QualityReport


def fake_predict(img):
    blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    ok = blur > 200
    p = Prediction(1, 0.99 if ok else 0.4, 5.0, 10.0, 1.0, 360, [])
    q = QualityReport(ok, blur, 80.0, 0.3, [] if ok else ["imagen desenfocada"])
    d = Decision("PICK" if ok else "DIVERT", 1, 10.0, "exact", 0.99 if ok else 0.1, q.reasons)
    return p, q, d


@pytest.fixture(scope="module")
def conn():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(20, 128, 128, 3), dtype=np.uint8)  # ruido = muy nítido
    index = pd.DataFrame({"obj_id": [1] * 20, "angle": [10] * 20})
    c = connect(":memory:")
    init_schema(c)
    seed_state(c, "P1", "L1", 0.9, "1.0.0")
    simulate(c, fake_predict, images, index, datetime(2026, 10, 26, 6), "P1", "L1")
    simulate(c, fake_predict, images, index, datetime(2026, 10, 26, 6), "P2", "L1", rate_per_min=0.1)
    return c


def test_three_shifts(conn):
    shifts = {r["shift"] for r in rows(conn, "SELECT DISTINCT shift FROM inspections WHERE plant_id=?", ("P1",))}
    assert shifts == {1, 2, 3}


def test_3am_incident_raises_diverts(conn):
    r = rows(conn, """SELECT substr(ts,12,2) AS h, AVG(action='DIVERT') AS rate FROM inspections
                      WHERE plant_id=? GROUP BY h""", ("P1",))
    rate = {x["h"]: x["rate"] for x in r}
    others = np.mean([v for h, v in rate.items() if h not in ("02", "03")])
    assert rate["03"] >= 2 * others and rate["03"] > 0.5


def test_vacuum_failures_at_17h(conn):
    r = rows(conn, "SELECT substr(ts,12,2) AS h, COUNT(*) AS n FROM grasp_events WHERE failure_code='VACUUM_LOSS' "
                   "AND plant_id=? GROUP BY h", ("P1",))
    assert r and all(x["h"] in ("17", "18") for x in r)


def test_parametrized_queries_survive_injection(conn):
    r = rows(conn, "SELECT COUNT(*) AS n FROM inspections WHERE plant_id=?", ("P1'; DROP TABLE inspections;--",))
    assert r[0]["n"] == 0
    assert rows(conn, "SELECT COUNT(*) AS n FROM inspections")[0]["n"] > 0


def test_second_plant_exists_for_isolation(conn):
    assert rows(conn, "SELECT COUNT(*) AS n FROM inspections WHERE plant_id='P2'")[0]["n"] > 0
