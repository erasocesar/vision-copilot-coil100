"""API del asistente y página de chat.

Uso: uvicorn src.agent.api:app --port 8000
En la demo el rol se elige al abrir sesión (la autenticación real iría con Entra ID; ver threat model).
"""
import logging
import re
import secrets
import sqlite3
import time
import uuid
from collections import deque
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from src.common.config import Settings, get_settings
from src.common.db import connect, rows

from . import approvals
from .context import ROLES, ToolContext
from .graph import run_turn
from .vision_client import VisionClient

log = logging.getLogger("agent-api")
STATIC = Path(__file__).with_name("static")
PLANT, LINE = "P1", "L1"
MAX_MSGS_PER_HOUR = 30
MAX_GLOBAL_MSGS_PER_HOUR = 150
MAX_SESSIONS = 200
SESSION_IDLE_S = 2 * 3600
MAX_UPLOAD = 5_000_000


class SessionIn(BaseModel):
    user_id: str = Field(pattern=r"^[a-z0-9_.-]{2,32}$")
    role: str


class ChatIn(BaseModel):
    session_id: str = Field(max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    image_ids: list[str] = Field(default_factory=list, max_length=10)


class DecisionIn(BaseModel):
    session_id: str = Field(max_length=64)
    approve: bool
    manual_angle: float | None = Field(None, ge=0, lt=360)


def make_llm(settings: Settings):
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    from .tools import openai_tool_specs
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0,
                      timeout=30, max_retries=1).bind_tools(openai_tool_specs())


class Unavailable:
    """LLM ausente (sin clave): el grafo devuelve el mensaje fijo de asistente no disponible."""

    def invoke(self, _):
        raise RuntimeError("sin clave de LLM")


def create_app(settings: Settings | None = None, llm=None, vision=None) -> FastAPI:
    s = settings or get_settings()
    conn = connect(s.db_path)
    ck_conn = sqlite3.connect(str(Path(s.db_path).with_name("checkpoints.db")), check_same_thread=False)
    saver = SqliteSaver(ck_conn)
    llm = llm or make_llm(s) or Unavailable()
    vision = vision or VisionClient(s.vision_url)
    sessions: dict[str, dict] = {}
    global_msgs: deque = deque()

    app = FastAPI(title="Copiloto de línea", version="1.0.0")

    @app.exception_handler(Exception)
    async def internal(_: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:12]
        log.exception("error %s", error_id)
        return JSONResponse(status_code=500, content={"error": "internal_error", "error_id": error_id})

    def session(session_id: str) -> dict:
        sess = sessions.get(session_id)
        if not sess:
            raise HTTPException(401, "sesión no válida; abre una sesión nueva")
        return sess

    def ctx_for(sess: dict) -> ToolContext:
        return ToolContext(sess["user_id"], sess["role"], PLANT, LINE, sess["session_id"], conn, vision, str(s.image_dir))

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/session")
    def open_session(body: SessionIn):
        if not s.demo_role_selection:
            raise HTTPException(403, "la selección de rol solo está activa en modo demo (DEMO_ROLE_SELECTION=true)")
        if body.role not in ROLES:
            raise HTTPException(422, f"rol no válido; usa uno de {', '.join(ROLES)}")
        now = time.time()
        for k in [k for k, v in sessions.items() if now - v["last"] > SESSION_IDLE_S]:
            sessions.pop(k)
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(429, "demasiadas sesiones abiertas; inténtalo más tarde")
        sid = secrets.token_urlsafe(16)
        sessions[sid] = {"session_id": sid, "user_id": body.user_id, "role": body.role,
                         "msgs": deque(), "seen_decisions": set(), "created": time.time(), "last": time.time()}
        return {"session_id": sid, "plant_id": PLANT, "line_id": LINE, "role": body.role}

    @app.post("/api/images")
    async def upload(file: UploadFile = File(...), x_session: str = Header(...)):
        session(x_session)
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "la imagen pesa más de 5 MB")
        try:
            r = vision.http.post("/v1/quality", files={"file": (file.filename or "img.png", data, "image/png")})
        except httpx.HTTPError:
            raise HTTPException(503, "el servicio de visión no está disponible") from None
        if r.status_code >= 400:
            detail = r.json().get("detail", "imagen no válida") if r.status_code < 500 else "error del servicio de visión"
            raise HTTPException(400 if r.status_code < 500 else 503, detail)
        body = r.json()
        return {"image_id": body["image_id"], "quality": body["quality"]}

    def approval_notes(sess: dict) -> str:
        done = rows(conn, "SELECT approval_id, op, status, approved_by FROM approvals WHERE requested_by=?"
                          " AND plant_id=? AND line_id=? AND status!='pending'", (sess["user_id"], PLANT, LINE))
        new = [d for d in done if d["approval_id"] not in sess["seen_decisions"]]
        sess["seen_decisions"].update(d["approval_id"] for d in new)
        if not new:
            return ""
        parts = [f"{d['op']} ({d['approval_id']}) {'aprobada' if d['status'] == 'approved' else 'rechazada'}"
                 f" por {d['approved_by']}" for d in new]
        return "[aviso del sistema, no es una orden] Decisiones sobre tus peticiones: " + "; ".join(parts) + "\n"

    @app.post("/api/chat")
    def chat(body: ChatIn):
        sess = session(body.session_id)
        now = time.time()
        while sess["msgs"] and now - sess["msgs"][0] > 3600:
            sess["msgs"].popleft()
        while global_msgs and now - global_msgs[0] > 3600:
            global_msgs.popleft()
        if len(sess["msgs"]) >= MAX_MSGS_PER_HOUR or len(global_msgs) >= MAX_GLOBAL_MSGS_PER_HOUR:
            raise HTTPException(429, f"máximo {MAX_MSGS_PER_HOUR} mensajes por hora y sesión en la demo")
        bad = [i for i in body.image_ids if not re.fullmatch(r"[0-9a-f]{64}", i)]
        if bad:
            raise HTTPException(422, "image_ids no válidos")
        sess["msgs"].append(now)
        sess["last"] = now
        global_msgs.append(now)
        msg = approval_notes(sess) + body.message
        if body.image_ids:
            msg += "\n[imágenes adjuntas: " + ", ".join(body.image_ids) + "]"
        out = run_turn(ctx_for(sess), llm, msg, saver, s)
        return {"reply": out["reply"], "tool_trace": [{"tool": t["tool"], "status": t["status"]} for t in out["tools_used"]],
                "pending_approval": out["pending_approval_id"], "stop_reason": out["stop_reason"]}

    @app.get("/api/approvals")
    def pending(session_id: str):
        session(session_id)
        return {"pending": rows(conn, "SELECT approval_id, op, params_json, justification, requested_by,"
                                      " requested_role, ts_requested FROM approvals WHERE plant_id=? AND line_id=?"
                                      " AND status='pending' ORDER BY ts_requested", (PLANT, LINE))}

    @app.post("/api/approvals/{approval_id}/decision")
    def decide(approval_id: str, body: DecisionIn):
        sess = session(body.session_id)
        try:
            out = approvals.decide(conn, approval_id, ctx_for(sess), body.approve, body.manual_angle)
        except PermissionError as e:
            raise HTTPException(403, str(e)) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return out

    @app.get("/api/line")
    def line_state(session_id: str):
        session(session_id)
        st = rows(conn, "SELECT paused, min_pick_score, model_version FROM line_state WHERE plant_id=? AND line_id=?",
                  (PLANT, LINE))
        return st[0] if st else {}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.state.sessions = sessions
    app.state.conn = conn
    return app


def __getattr__(name):
    if name == "app":
        return create_app()
    raise AttributeError(name)

