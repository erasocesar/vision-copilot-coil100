"""Guardia determinista antes de cada llamada a una tool: existencia, esquema, rol y límites del agente."""
from dataclasses import dataclass

from pydantic import ValidationError

from .audit import args_hash
from .tools import TOOLS

MAX_TOOL_CALLS = 8
MAX_IDENTICAL = 2  # la tercera llamada idéntica en el mismo turno se corta como bucle


@dataclass
class GuardResult:
    ok: bool
    error: dict | None = None
    stop: bool = False  # True: el turno debe terminar (bucle o límite)


def check_call(ctx, name: str, args: dict, turn_calls: list[tuple[str, str]]) -> GuardResult:
    if len(turn_calls) >= MAX_TOOL_CALLS:
        return GuardResult(False, {"error": "max_tool_calls", "detail": f"máximo {MAX_TOOL_CALLS} llamadas por turno"}, True)
    key = (name, args_hash(args or {}))
    if turn_calls.count(key) >= MAX_IDENTICAL:
        return GuardResult(False, {"error": "loop_detected", "detail": f"{name} repetida con los mismos argumentos"}, True)
    tool = TOOLS.get(name)
    if tool is None:
        return GuardResult(False, {"error": "unknown_tool", "detail": f"no existe la tool {name}"})
    try:
        tool.input_model(**(args or {}))
    except ValidationError as e:
        err = e.errors()[0]
        return GuardResult(False, {"error": "invalid_args", "detail": f"{'.'.join(map(str, err['loc']))}: {err['msg']}"})
    if ctx.role not in tool.roles:
        return GuardResult(False, {"error": "forbidden", "detail": f"el rol {ctx.role} no puede usar {name}"})
    return GuardResult(True)
