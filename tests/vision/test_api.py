from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.common.config import Settings
from src.vision.api import create_app

SAMPLE = Path("data/raw/obj1__0.png")


def _client(model_dir, tmp_path, faults: bool) -> TestClient:
    s = Settings(model_dir=model_dir, image_dir=tmp_path / "img", demo_faults=faults)
    return TestClient(create_app(s), raise_server_exceptions=False)


@pytest.fixture
def client(tiny_model_dir, tmp_path):
    return _client(tiny_model_dir, tmp_path, faults=True)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_version"] == "test"


def test_predict_known_object(client):
    r = client.post("/v1/predict", files={"file": ("obj1__0.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["obj_id"] == 1
    assert body["decision"]["action"] in {"PICK", "DIVERT"}
    assert len(body["image_id"]) == 64 and body["latency_ms"] >= 0
    assert len(body["prediction"]["topk"]) == 5


def test_quality_does_not_run_model(client):
    r = client.post("/v1/quality", files={"file": ("x.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 200 and "prediction" not in r.json() and r.json()["quality"]["ok"] in (True, False)


def test_invalid_image_400(client):
    r = client.post("/v1/predict", files={"file": ("x.txt", b"hola", "text/plain")})
    assert r.status_code == 400 and r.json()["error"] == "invalid_image"


def test_top_k_limit(client):
    r = client.post("/v1/predict?top_k=50", files={"file": ("x.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 422


def test_fault_injection_enabled(client):
    r = client.post("/v1/predict", headers={"X-Demo-Fault": "500"},
                    files={"file": ("x.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 503


def test_fault_injection_disabled(tiny_model_dir, tmp_path):
    c = _client(tiny_model_dir, tmp_path, faults=False)
    r = c.post("/v1/predict", headers={"X-Demo-Fault": "500"},
               files={"file": ("x.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 200


def test_500_hides_traceback(client, monkeypatch):
    import src.vision.api as api
    monkeypatch.setattr(api, "decode_image", lambda b: (_ for _ in ()).throw(RuntimeError("boom secreto")))
    r = client.post("/v1/predict", files={"file": ("x.png", SAMPLE.read_bytes(), "image/png")})
    assert r.status_code == 500 and "Traceback" not in r.text and "secreto" not in r.text
    assert "error_id" in r.json()
