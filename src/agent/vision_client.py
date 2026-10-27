"""Cliente HTTP del servicio de visión: reintentos acotados y corte de circuito."""
import time
from pathlib import Path

import httpx

RETRY_WAITS = (0.5, 1.0)
BREAKER_FAILURES = 3
BREAKER_OPEN_S = 30.0


class VisionError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind  # timeout | server | bad_request | circuit_open
        self.detail = detail


class VisionClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep, clock=time.monotonic):
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self.sleep, self.clock = sleep, clock
        self.failures = 0
        self.open_until = 0.0
        self.fault: str | None = None  # solo demo/evaluación: se envía como X-Demo-Fault

    def predict(self, image_path: Path, top_k: int = 3) -> dict:
        return self._post("/v1/predict", image_path, {"top_k": top_k})

    def quality(self, image_path: Path) -> dict:
        return self._post("/v1/quality", image_path, {})

    def _post(self, url: str, image_path: Path, params: dict) -> dict:
        if self.clock() < self.open_until:
            raise VisionError("circuit_open", "servicio de visión en pausa tras fallos seguidos")
        data = Path(image_path).read_bytes()
        headers = {"X-Demo-Fault": self.fault} if self.fault else {}
        last = VisionError("server", "sin respuesta")
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                r = self.http.post(url, params=params, headers=headers,
                                   files={"file": (Path(image_path).name, data, "image/png")})
            except httpx.TimeoutException:
                last = VisionError("timeout", "el servicio de visión no respondió a tiempo")
            except httpx.HTTPError as e:
                last = VisionError("server", type(e).__name__)
            else:
                if r.status_code < 400:
                    self.failures = 0
                    return r.json()
                if r.status_code < 500:
                    is_json = r.headers.get("content-type", "").startswith("application/json")
                    body = r.json() if is_json else {}
                    raise VisionError("bad_request", str(body.get("detail") or body.get("error") or r.status_code))
                last = VisionError("server", f"HTTP {r.status_code}")
            if attempt < len(RETRY_WAITS):
                self.sleep(RETRY_WAITS[attempt])
        self.failures += 1
        if self.failures >= BREAKER_FAILURES:
            self.open_until = self.clock() + BREAKER_OPEN_S
            self.failures = 0
        raise last
