"""Protocolos de partición A-D y validación interna sin tocar el conjunto de prueba."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ARC_VIEWS = 12  # 60° contiguos = 12 vistas de 5°
N_UNKNOWN = 10


@dataclass
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict = field(default_factory=dict)


def _idx(mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.to_numpy())


def _arc(start: int, n: int = ARC_VIEWS) -> set[int]:
    return {(start + 5 * i) % 360 for i in range(n)}


def make_split(index: pd.DataFrame, protocol: Literal["A", "B", "C", "D"], seed: int = 42) -> Split:
    rng = np.random.default_rng(seed)
    a = index["angle"]
    if protocol == "A":
        return Split("A", _idx(a % 10 == 0), _idx(a % 10 == 5))
    if protocol == "B":
        starts = {int(o): int(rng.integers(0, 72)) * 5 for o in sorted(index["obj_id"].unique())}
        in_arc = pd.Series([ang in _arc(starts[o]) for o, ang in zip(index["obj_id"], a)], index=index.index)
        return Split("B", _idx(~in_arc), _idx(in_arc), {"arc_start": starts})
    if protocol == "C":
        return Split("C", _idx(a % 30 == 0), _idx(a % 30 != 0))
    if protocol == "D":
        objs = sorted(index["obj_id"].unique())
        unknown = sorted(int(o) for o in rng.choice(objs, N_UNKNOWN, replace=False))
        is_unk = index["obj_id"].isin(unknown)
        train = ~is_unk & (a % 10 == 0)
        test = is_unk | (~is_unk & (a % 10 == 5))
        return Split("D", _idx(train), _idx(test), {"unknown_objs": unknown})
    raise ValueError(f"protocolo desconocido: {protocol}")


def make_validation(index: pd.DataFrame, split: Split, seed: int = 42) -> Split | list[Split]:
    """Validación que imita el salto de cada protocolo usando solo las vistas de entrenamiento."""
    rng = np.random.default_rng(seed + 1)
    tr = index.iloc[split.train_idx]
    a = tr["angle"]
    pos = split.train_idx

    if split.name == "A":
        return Split("A-val", pos[(a % 20 == 0).to_numpy()], pos[(a % 20 == 10).to_numpy()])

    if split.name == "B":
        # Otro arco ciego de 12 vistas, completamente dentro de las 60 vistas vistas.
        starts = {}
        for o, s in split.meta["arc_start"].items():
            first_seen = (s + 5 * ARC_VIEWS) % 360
            offset = int(rng.integers(0, 60 - ARC_VIEWS + 1))
            starts[int(o)] = (first_seen + 5 * offset) % 360
        in_arc = np.array([ang in _arc(starts[o]) for o, ang in zip(tr["obj_id"], a)])
        return Split("B-val", pos[~in_arc], pos[in_arc], {"arc_start": starts})

    if split.name == "C":
        folds = []
        for k, held in enumerate(range(0, 360, 30)):
            m = (a == held).to_numpy()
            folds.append(Split(f"C-val{k}", pos[~m], pos[m], {"held_angle": held}))
        return folds

    if split.name == "D":
        known = sorted(tr["obj_id"].unique())
        sim_unknown = sorted(int(o) for o in rng.choice(known, N_UNKNOWN, replace=False))
        is_unk = tr["obj_id"].isin(sim_unknown).to_numpy()
        a_np = a.to_numpy()
        train = ~is_unk & (a_np % 20 == 0)
        test = is_unk | (~is_unk & (a_np % 20 == 10))
        return Split("D-val", pos[train], pos[test], {"unknown_objs": sim_unknown})

    raise ValueError(split.name)


def save_splits(splits: dict[str, Split], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        meta = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in s.meta.items()}
        (out_dir / f"{name}.json").write_text(json.dumps({
            "name": s.name, "train_idx": s.train_idx.tolist(), "test_idx": s.test_idx.tolist(), "meta": meta,
        }), encoding="utf-8")


def load_split(path: Path) -> Split:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = {k: ({int(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in d["meta"].items()}
    return Split(d["name"], np.array(d["train_idx"], dtype=int), np.array(d["test_idx"], dtype=int), meta)
