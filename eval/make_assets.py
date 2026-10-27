"""Genera las imágenes de la evaluación del agente en eval/assets/ (a partir de COIL-100 y sintéticas).

Uso: python eval/make_assets.py --raw data/raw --model models/vision/v1.0.0
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path("eval/assets")


def read(raw: Path, obj: int, angle: int) -> np.ndarray:
    return cv2.imread(str(raw / f"obj{obj}__{angle}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--model", type=Path, default=Path("models/vision/v1.0.0"))
    args = ap.parse_args()
    unknown = json.loads((args.model / "manifest.json").read_text(encoding="utf-8"))["unknown_objs"]
    OUT.mkdir(parents=True, exist_ok=True)

    # Vistas intermedias (5°) de objetos conocidos: el modelo nunca las vio.
    for obj, ang in [(5, 45), (12, 125), (47, 215), (67, 95), (30, 305), (1, 15)]:
        cv2.imwrite(str(OUT / f"known_obj{obj}_{ang}.png"), read(args.raw, obj, ang))
    # Objeto desconocido (excluido del entrenamiento).
    u = unknown[0]
    cv2.imwrite(str(OUT / f"unknown_obj{u}_0.png"), read(args.raw, u, 0))
    # Vista desenfocada: resultado no confiable.
    cv2.imwrite(str(OUT / "blurred_obj5_45.png"), cv2.GaussianBlur(read(args.raw, 5, 45), (0, 0), 4))
    # Fuera de dominio: escena sintética sin fondo negro ni objeto aislado.
    rng = np.random.default_rng(0)
    scene = np.zeros((128, 128, 3), np.uint8)
    scene[:] = (180, 200, 220)
    for _ in range(12):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        cv2.rectangle(scene, tuple(int(x) for x in rng.integers(0, 128, 2)), tuple(int(x) for x in rng.integers(0, 128, 2)), c, -1)
    cv2.imwrite(str(OUT / "ood_scene.png"), scene)
    # Producto nuevo para el alta: vistas cada 10° (suficiente) y cada 30° (insuficiente).
    new = unknown[1]
    sku_dir = OUT / f"sku_obj{new}"
    sku_dir.mkdir(exist_ok=True)
    for ang in range(0, 360, 10):
        cv2.imwrite(str(sku_dir / f"view_{ang:03d}.png"), read(args.raw, new, ang))
    (OUT / "assets.json").write_text(json.dumps({"unknown_obj": u, "sku_obj": new}), encoding="utf-8")
    print("assets en", OUT, "desconocido", u, "sku", new)


if __name__ == "__main__":
    main()
