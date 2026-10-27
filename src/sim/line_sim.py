"""Simulador de línea: 24 horas de inspecciones con el modelo real, para que el asistente tenga datos.

Incidentes sembrados a propósito (el agente debe encontrarlos cruzando datos):
- 02:30-04:00: la cámara pierde foco; sube el desenfoque y con él los desvíos a revisión.
- 17:00-18:00: fallo mecánico de la ventosa del robot (agarres fallidos con la visión correcta).

Uso: python -m src.sim.line_sim --db appdata/line.db --model models/vision/v1.0.0
"""
import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from src.common.db import connect, init_schema
from src.vision.circular import sym_dist

PredictFn = Callable[[np.ndarray], tuple]  # img -> (Prediction, QualityReport, Decision)


def shift_of(ts: datetime) -> int:
    h = ts.hour
    return 1 if 6 <= h < 14 else (2 if 14 <= h < 22 else 3)


def blur_sigma(ts: datetime, incident_day: datetime) -> float:
    """Desenfoque de la cámara: crece de 02:30 a 03:30 y se corrige a las 04:00."""
    start = incident_day.replace(hour=2, minute=30)
    peak = incident_day.replace(hour=3, minute=30)
    end = incident_day.replace(hour=4, minute=0)
    if start <= ts < peak:
        return 0.5 + 2.5 * (ts - start) / (peak - start)
    if peak <= ts < end:
        return 3.0
    return 0.0


def simulate(conn, predict: PredictFn, images: np.ndarray, index: pd.DataFrame, start: datetime,
             plant: str = "P1", line: str = "L1", rate_per_min: float = 1.0, seed: int = 42,
             model_version: str = "1.0.0", image_dir: Path | None = None, hours: int = 24) -> dict:
    rng = np.random.default_rng(seed)
    incident_day = (start + timedelta(days=1)).replace(hour=0, minute=0)
    step = timedelta(minutes=1 / rate_per_min)
    n = int(hours * 60 * rate_per_min)
    counts = {"inspections": 0, "picks": 0, "diverts": 0, "grasp_fail": 0}
    insp_rows, cam_rows, robot_rows, grasp_rows, queue_rows = [], [], [], [], []

    for i in range(n):
        ts = start + i * step
        k = int(rng.integers(0, len(index)))
        img = images[k]
        sigma = blur_sigma(ts, incident_day)
        if sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        pred, q, dec = predict(img)
        true_obj, true_angle = int(index.obj_id.iloc[k]), int(index.angle.iloc[k])
        image_id = hashlib.sha256(img.tobytes()).hexdigest()
        insp_id = i + 1 + (0 if plant == "P1" else 1_000_000)
        insp_rows.append((insp_id, plant, line, ts.isoformat(), shift_of(ts), image_id, true_obj, true_angle,
                          pred.obj_id, pred.class_prob, pred.ood_distance, pred.angle, pred.angle_std,
                          dec.angle_mode, dec.pick_score, dec.action, json.dumps(dec.reasons, ensure_ascii=False),
                          model_version, q.blur))
        robot_rows.append((plant, line, ts.isoformat(), insp_id, dec.action, dec.obj_id, dec.angle, dec.angle_mode))
        counts["inspections"] += 1
        if dec.action == "PICK":
            counts["picks"] += 1
            wrong_sku = pred.obj_id != true_obj
            misaligned = not wrong_sku and float(sym_dist(dec.angle, true_angle, pred.symmetry)) > 10
            mech = 17 <= ts.hour < 18 and rng.random() < 0.35
            code = "WRONG_SKU" if wrong_sku else "MISALIGNED" if misaligned else "VACUUM_LOSS" if mech else None
            grasp_rows.append((plant, line, (ts + timedelta(seconds=8)).isoformat(), insp_id, int(code is None), code))
            counts["grasp_fail"] += int(code is not None)
        else:
            counts["diverts"] += 1
            queue_rows.append((f"{plant}-{line}-{insp_id}", plant, line, insp_id, "pending", ts.isoformat()))
            if image_dir is not None:
                image_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_dir / f"{image_id}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % max(1, int(5 * rate_per_min)) == 0:
            cam_rows.append((plant, line, ts.isoformat(), q.blur, q.brightness,
                             int(rng.poisson(0.2 + 6 * (sigma > 0))), float(38 + rng.normal(0, 0.8) + 4 * (sigma > 0))))

    conn.executemany("INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", insp_rows)
    conn.executemany("INSERT INTO robot_commands (plant_id,line_id,ts,inspection_id,action,obj_id,angle,angle_mode)"
                     " VALUES (?,?,?,?,?,?,?,?)", robot_rows)
    conn.executemany("INSERT INTO grasp_events (plant_id,line_id,ts,inspection_id,success,failure_code)"
                     " VALUES (?,?,?,?,?,?)", grasp_rows)
    conn.executemany("INSERT INTO review_queue VALUES (?,?,?,?,?,?)", queue_rows)
    conn.executemany("INSERT INTO camera_health VALUES (?,?,?,?,?,?,?)", cam_rows)
    conn.commit()
    return counts


def seed_state(conn, plant: str, line: str, min_pick_score: float, model_version: str) -> None:
    conn.execute("INSERT OR REPLACE INTO line_state VALUES (?,?,0,?,?)", (plant, line, min_pick_score, model_version))
    conn.execute("INSERT OR IGNORE INTO model_versions VALUES (?, 'production', '{}')", (model_version,))
    conn.execute("INSERT OR IGNORE INTO model_versions VALUES ('1.1.0-rc1', 'staging', ?)",
                 (json.dumps({"nota": "candidata con más vistas del producto 12"}),))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/line.db"))
    ap.add_argument("--model", type=Path, default=Path("models/vision/v1.0.0"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--image-dir", type=Path, default=Path("appdata/images"))
    ap.add_argument("--start", default="2026-10-26T06:00:00")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from src.vision.data import load_images, load_index
    from src.vision.model import VisionModel
    from src.vision.policy import Thresholds, decide
    from src.vision.quality import QualityRefs, check_quality

    if args.db.exists() and not args.force:
        print(f"{args.db} ya existe; usa --force para regenerarla")
        return
    if args.db.exists():
        args.db.unlink()
    model = VisionModel.load(args.model)
    th_raw = json.loads((args.model / "thresholds.json").read_text(encoding="utf-8"))
    th, refs = Thresholds.from_dict(th_raw), QualityRefs(**th_raw["quality"])
    version = json.loads((args.model / "manifest.json").read_text(encoding="utf-8"))["version"]

    def predict(img):
        q = check_quality(img, refs)
        p = model.predict(img[None], top_k=3)[0]
        return p, q, decide(p, q, th)

    index = load_index(args.raw)
    known = set(int(o) for o in model.obj_ids)
    # vistas que el modelo no vio: las intermedias de los conocidos y, de vez en cuando, un desconocido
    pool = index[(index.obj_id.isin(known) & (index.angle % 10 == 5)) |
                 (~index.obj_id.isin(known) & (index.angle % 60 == 0))].reset_index(drop=True)
    images = load_images(pool)
    conn = connect(args.db)
    init_schema(conn)
    seed_state(conn, "P1", "L1", th.min_pick_score, version)
    seed_state(conn, "P2", "L1", th.min_pick_score, version)
    start = datetime.fromisoformat(args.start)
    c1 = simulate(conn, predict, images, pool, start, "P1", "L1", model_version=version, image_dir=args.image_dir)
    c2 = simulate(conn, predict, images, pool, start, "P2", "L1", rate_per_min=0.1, seed=7, model_version=version)
    print("P1:", c1, "| P2:", c2)


if __name__ == "__main__":
    main()
