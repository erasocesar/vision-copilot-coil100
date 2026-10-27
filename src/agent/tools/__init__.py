"""Catálogo de tools del agente.

Cada tool declara su contrato (Pydantic), los roles que pueden usarla y si tiene efectos.
`run_tool` es el único punto de entrada: valida, comprueba permisos, ejecuta, recorta la
salida para el LLM y deja la llamada en auditoría.
"""
import json
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from ..audit import audit
from ..context import ROLES, ToolContext
from ..vision_client import VisionError

ALL = set(ROLES)


class ToolError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code, self.detail = code, detail


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    roles: set[str]
    side_effect: bool
    run: Callable[[ToolContext, BaseModel], dict]


TOOLS: dict[str, Tool] = {}


def register(name: str, description: str, input_model: type[BaseModel], roles: set[str] = ALL,
             side_effect: bool = False):
    def deco(fn):
        TOOLS[name] = Tool(name, description, input_model, roles, side_effect, fn)
        return fn
    return deco


MAX_OUTPUT_CHARS = 4000


def run_tool(ctx: ToolContext, name: str, args: dict) -> dict:
    """Ejecuta una tool y devuelve siempre un dict (resultado o error estructurado)."""
    tool = TOOLS.get(name)
    if tool is None:
        audit(ctx, name, args, "error", "unknown_tool")
        return {"error": "unknown_tool", "detail": f"no existe la tool {name}"}
    if ctx.role not in tool.roles:
        audit(ctx, name, args, "forbidden")
        return {"error": "forbidden", "detail": f"el rol {ctx.role} no puede usar {name}"}
    try:
        parsed = tool.input_model(**(args or {}))
    except ValidationError as e:
        err = e.errors()[0]
        audit(ctx, name, args, "error", "invalid_args")
        return {"error": "invalid_args", "detail": f"{'.'.join(map(str, err['loc']))}: {err['msg']}"}
    try:
        out = tool.run(ctx, parsed)
    except ToolError as e:
        audit(ctx, name, args, "error", e.code)
        return {"error": e.code, "detail": e.detail}
    except PermissionError as e:
        audit(ctx, name, args, "forbidden", str(e))
        return {"error": "forbidden", "detail": str(e)}
    except VisionError as e:
        audit(ctx, name, args, "error", f"vision_{e.kind}")
        return {"error": f"vision_{e.kind}", "detail": e.detail,
                "hint": "no inventes el resultado; informa al usuario y propone reintentar o escalar"}
    except ValueError as e:
        audit(ctx, name, args, "error", "invalid_request")
        return {"error": "invalid_request", "detail": str(e)}
    audit(ctx, name, args, "ok")
    text = json.dumps(out, ensure_ascii=False, default=str)
    if len(text) > MAX_OUTPUT_CHARS:
        out = {"truncated": True, "data": text[:MAX_OUTPUT_CHARS]}
    return out


def openai_tool_specs() -> list[dict]:
    """Esquemas para bind_tools (formato de funciones de OpenAI)."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.input_model.model_json_schema()}}
            for t in TOOLS.values()]


# Registro de las 13 tools (el import las añade a TOOLS).
from . import image, maintenance, ops, review, risk, sku  # noqa: E402,F401
