import numpy as np
import pandas as pd
import pytest

from src.vision.circular import circ_dist
from src.vision.protocols import load_split, make_split, make_validation, save_splits


@pytest.fixture(scope="module")
def index():
    rows = [(o, a, f"obj{o}__{a}.png") for o in range(1, 101) for a in range(0, 360, 5)]
    return pd.DataFrame(rows, columns=["obj_id", "angle", "path"])


def test_A_counts(index):
    s = make_split(index, "A")
    assert len(s.train_idx) == 3600 and len(s.test_idx) == 3600
    assert (index.loc[s.train_idx, "angle"] % 10 == 0).all()
    assert (index.loc[s.test_idx, "angle"] % 10 == 5).all()


def test_B_arc_contiguous(index):
    s = make_split(index, "B")
    test = index.loc[s.test_idx]
    assert len(s.train_idx) == 6000 and len(test) == 1200
    for obj, g in test.groupby("obj_id"):
        start = s.meta["arc_start"][int(obj)]
        expected = sorted((start + 5 * i) % 360 for i in range(12))
        assert sorted(g["angle"]) == expected


def test_C_counts(index):
    s = make_split(index, "C")
    assert len(s.train_idx) == 1200 and len(s.test_idx) == 6000
    assert (index.loc[s.train_idx, "angle"] % 30 == 0).all()


def test_D_unknown_absent_from_train(index):
    s = make_split(index, "D")
    unknown = set(s.meta["unknown_objs"])
    assert len(unknown) == 10
    assert not unknown & set(index.loc[s.train_idx, "obj_id"])
    test = index.loc[s.test_idx]
    assert (test[test.obj_id.isin(unknown)].groupby("obj_id").size() == 72).all()
    assert test.obj_id.nunique() == 100


@pytest.mark.parametrize("protocol", ["A", "B", "C", "D"])
def test_validation_never_touches_test(index, protocol):
    s = make_split(index, protocol)
    vals = make_validation(index, s)
    vals = vals if isinstance(vals, list) else [vals]
    test = set(s.test_idx)
    for v in vals:
        assert not set(v.train_idx) & test
        assert not set(v.test_idx) & test
        assert not set(v.train_idx) & set(v.test_idx)
        assert len(v.test_idx) > 0


def test_C_validation_is_leave_one_view_out(index):
    vals = make_validation(index, make_split(index, "C"))
    assert isinstance(vals, list) and len(vals) == 12


def test_B_validation_arc_inside_seen_views(index):
    s = make_split(index, "B")
    v = make_validation(index, s)
    for obj, g in index.loc[v.test_idx].groupby("obj_id"):
        angles = sorted(g["angle"])
        assert len(angles) == 12
        # contiguo: 11 saltos de 5° en orden circular
        gaps = sorted(circ_dist(np.array(angles), np.roll(angles, 1)))
        assert gaps[:11] == [5] * 11


def test_seed_reproducible(index, tmp_path):
    a, b = make_split(index, "D", seed=42), make_split(index, "D", seed=42)
    assert a.meta == b.meta and np.array_equal(a.test_idx, b.test_idx)
    save_splits({"D": a}, tmp_path)
    c = load_split(tmp_path / "D.json")
    assert np.array_equal(c.test_idx, a.test_idx) and c.meta["unknown_objs"] == a.meta["unknown_objs"]
