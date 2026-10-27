"""Registro de auditoría: quién llamó qué tool y con qué resultado. Los argumentos se guardan como hash."""
import hashlib
import json

from src.common.db import now_iso


def args_hash(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def audit(ctx, tool: str, args: dict, status: str, detail: str = "") -> None:
    ctx.conn.execute(
        "INSERT INTO audit (ts, session_id, user_id, role, plant_id, line_id, tool, args_sha256, status, detail)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (now_iso(), ctx.session_id, ctx.user_id, ctx.role, ctx.plant_id, ctx.line_id, tool,
         args_hash(args), status, detail[:500]))
    ctx.conn.commit()
