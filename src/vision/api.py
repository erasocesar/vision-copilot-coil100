"""Servicio de inferencia de visión (independiente del agente).

Uso: uvicorn src.vision.api:app --port 8001
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.common.config import Settings, get_settings

from .model import VisionModel
from .policy import Thresholds, decide
from .sku import evaluate_views
from .quality import MAX_BYTES, InvalidImage, QualityRefs, check_quality, decode_image

log = logging.getLogger("vision-api")


class QualityOut(BaseModel):
    ok: bool
    blur: float
    brightness: float
    fg_ratio: float
    reasons: list[str]


class NeighborOut(BaseModel):
    obj_id: int
    angle: int
    distance: float
    path: str


class PredictionOut(BaseModel):
    obj_id: int | None
    class_prob: float
    ood_distance: float
    angle: float | None
    angle_std: float
    symmetry: int
    topk: list[NeighborOut]


class DecisionOut(BaseModel):
    action: str
    obj_id: int | None
    angle: float | None
    angle_mode: str
    pick_score: float
    reasons: list[str]


class QualityResponse(BaseModel):
    image_id: str
    quality: QualityOut


class PredictResponse(BaseModel):
    image_id: str
    quality: QualityOut
    prediction: PredictionOut
    decision: DecisionOut
    model_version: str
    latency_ms: float


async def _read_limited(file: UploadFile) -> bytes:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise InvalidImage(f"la imagen pesa más de {MAX_BYTES // 1_000_000} MB")
    return data


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    model = VisionModel.load(s.model_dir)
    th_raw = json.loads((Path(s.model_dir) / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = Thresholds.from_dict(th_raw)
    refs = QualityRefs(**th_raw["quality"])
    manifest = json.loads((Path(s.model_dir) / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("version", "unknown")
    image_dir = Path(s.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Servicio de visión COIL-100", version=version,
                  description="Identificación de objeto, ángulo, vistas similares y decisión para el robot.")

    @app.exception_handler(InvalidImage)
    async def invalid_image(_: Request, exc: InvalidImage):
        return JSONResponse(status_code=400, content={"error": "invalid_image", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def internal(_: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:12]
        log.exception("error %s", error_id)
        return JSONResponse(status_code=500, content={"error": "internal_error", "error_id": error_id})

    async def demo_fault(fault: str | None):
        # Inyección de fallos solo para la demo de escenarios adversos.
        if not s.demo_faults or not fault:
            return None
        if fault == "500":
            return JSONResponse(status_code=503, content={"error": "service_unavailable", "detail": "fallo simulado"})
        if fault == "timeout":
            await asyncio.sleep(15)
        return None

    def _store(raw: bytes, img) -> str:
        image_id = hashlib.sha256(raw).hexdigest()
        path = image_dir / f"{image_id}.png"
        if not path.exists():
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return image_id

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": version}

    @app.get("/v1/models")
    def models():
        return manifest

    @app.post("/v1/quality", response_model=QualityResponse)
    async def quality(file: UploadFile = File(...), x_demo_fault: str | None = Header(default=None)):
        if (r := await demo_fault(x_demo_fault)) is not None:
            return r
        raw = await _read_limited(file)
        img = decode_image(raw)
        return {"image_id": _store(raw, img), "quality": asdict(check_quality(img, refs))}

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10),
                      x_demo_fault: str | None = Header(default=None)):
        if (r := await demo_fault(x_demo_fault)) is not None:
            return r
        t0 = time.perf_counter()
        raw = await _read_limited(file)
        img = decode_image(raw)
        q = check_quality(img, refs)
        pred = model.predict(img[None], top_k=top_k)[0]
        dec = decide(pred, q, thresholds)
        return {"image_id": _store(raw, img), "quality": asdict(q), "prediction": asdict(pred),
                "decision": asdict(dec), "model_version": version,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    @app.post("/v1/sku/evaluate")
    async def sku_evaluate(files: list[UploadFile] = File(...), angles: str = Form(...)):
        """Evalúa las vistas de un producto nuevo sin tocar el modelo en producción."""
        try:
            angle_list = [int(a) for a in angles.split(",")]
        except ValueError:
            raise InvalidImage("angles debe ser una lista de enteros separada por comas") from None
        if not 6 <= len(files) <= 72 or len(files) != len(angle_list):
            raise InvalidImage("se necesitan entre 6 y 72 vistas, una por ángulo")
        imgs, total = [], 0
        for f in files:
            raw = await _read_limited(f)
            total += len(raw)
            if total > 40_000_000:
                raise InvalidImage("la evaluación admite como máximo 40 MB en total")
            imgs.append(decode_image(raw))
        imgs = np.stack(imgs)
        return evaluate_views(model, imgs, angle_list, thresholds.max_ood)

    return app


def __getattr__(name):
    # `uvicorn src.vision.api:app` crea la app al pedirla, no al importar el módulo (los tests usan create_app).
    if name == "app":
        return create_app()
    raise AttributeError(name)
