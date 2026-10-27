"""Grafo del agente (LangGraph): un agente de tool-calling rodeado de nodos deterministas.

start -> (presupuesto) -> agent -> tools -> agent ... -> fin
- start: reinicia contadores del turno, recorta el historial y comprueba el tope de gasto.
- agent: única llamada al LLM; elige tools o responde.
- tools: guardia (esquema, rol, bucle, máximo de llamadas, tiempo) y ejecución auditada.
- finish: mensajes fijos cuando el turno se corta (presupuesto, LLM caído, bucle, límites).
"""
import json
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .audit import args_hash, audit
from .budget import budget_ok, record_usage
from .guard import check_call
from .prompts import system_prompt
from .tools import run_tool

TURN_TIMEOUT_S = 60
RECURSION_LIMIT = 20
KEEP_MESSAGES = 10
MAX_SUMMARY_CHARS = 800

STOP_MESSAGES = {
    "budget": "El asistente no está disponible ahora (se alcanzó el tope de gasto de la demo). "
              "El servicio de visión sigue funcionando.",
    "llm_unavailable": "El asistente no está disponible ahora. El servicio de visión sigue funcionando.",
    "loop_detected": "He detenido la consulta porque estaba repitiendo la misma operación sin avanzar. "
                     "Con lo obtenido no puedo darte una respuesta fiable. ¿Quieres que lo escale a un supervisor?",
    "max_tool_calls": "He llegado al máximo de pasos por consulta sin una respuesta fiable. "
                      "Divide la pregunta o deja que lo escale a un supervisor.",
    "timeout": "La consulta superó el tiempo máximo. Inténtalo de nuevo o deja que lo escale a un supervisor.",
}


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    summary: str
    turn_calls: list
    started_at: float
    stop_reason: str | None
    tools_used: list
    pending_approval_id: str | None


def _trim(messages: list, summary: str) -> tuple[list, str]:
    """Deja como mucho KEEP_MESSAGES mensajes, empezando por uno del usuario; lo quitado pasa al resumen."""
    if len(messages) <= KEEP_MESSAGES:
        return [], summary
    cut = len(messages) - KEEP_MESSAGES
    while cut < len(messages) and not isinstance(messages[cut], HumanMessage):
        cut += 1
    removed = messages[:cut]
    asked = [m.content[:120] for m in removed if isinstance(m, HumanMessage)]
    new_summary = (summary + " | " if summary else "") + "; ".join(asked)
    return [RemoveMessage(id=m.id) for m in removed], new_summary[-MAX_SUMMARY_CHARS:]


def build_graph(ctx, llm, checkpointer=None, settings=None):
    def start(state: AgentState) -> dict:
        removals, summary = _trim(state.get("messages", []), state.get("summary", ""))
        upd = {"turn_calls": [], "started_at": time.monotonic(), "stop_reason": None, "tools_used": [],
               "pending_approval_id": None, "summary": summary}
        if removals:
            upd["messages"] = removals
        if not budget_ok(ctx.conn, settings):
            upd["stop_reason"] = "budget"
        return upd

    def agent(state: AgentState) -> dict:
        # El tope se comprueba antes de cada llamada, no solo al empezar el turno.
        if not budget_ok(ctx.conn, settings):
            return {"stop_reason": "budget"}
        msgs =[SystemMessage(system_prompt(ctx, state.get("summary", "")))] + state["messages"]
        try:
            resp = llm.invoke(msgs)
        except Exception as e:  # noqa: BLE001 - cualquier fallo del proveedor se trata igual
            audit(ctx, "llm", {}, "error", type(e).__name__)
            return {"stop_reason": "llm_unavailable"}
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            record_usage(ctx.conn, ctx.session_id, getattr(resp, "response_metadata", {}).get("model_name", "llm"),
                         dict(usage), settings)
        return {"messages": [resp]}

    def tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = [tuple(c) for c in state.get("turn_calls", [])]
        used = list(state.get("tools_used", []))
        out_msgs, stop, pending = [], state.get("stop_reason"), state.get("pending_approval_id")
        for tc in last.tool_calls:
            name, args = tc["name"], tc.get("args") or {}
            if stop:
                result = {"error": "skipped", "detail": "turno detenido"}
            elif time.monotonic() - state["started_at"] > TURN_TIMEOUT_S:
                stop, result = "timeout", {"error": "timeout", "detail": "tiempo máximo del turno"}
            else:
                g = check_call(ctx, name, args, calls)
                if not g.ok:
                    result = g.error
                    audit(ctx, name, args, "blocked", g.error["error"])
                    if g.stop:
                        stop = g.error["error"]
                else:
                    result = run_tool(ctx, name, args)
                    calls.append((name, args_hash(args)))
                    if result.get("status") == "pending_approval":
                        pending = result["approval_id"]
            used.append({"tool": name, "args": args, "status": result.get("error", "ok")})
            out_msgs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str),
                                        tool_call_id=tc["id"], name=name))
        return {"messages": out_msgs, "turn_calls": [list(c) for c in calls], "tools_used": used,
                "stop_reason": stop, "pending_approval_id": pending}

    def finish(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=STOP_MESSAGES.get(state.get("stop_reason"), STOP_MESSAGES["llm_unavailable"]))]}

    def after_start(state: AgentState) -> str:
        return "finish" if state.get("stop_reason") else "agent"

    def after_agent(state: AgentState) -> str:
        if state.get("stop_reason"):
            return "finish"
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    def after_tools(state: AgentState) -> str:
        return "finish" if state.get("stop_reason") else "agent"

    g = StateGraph(AgentState)
    g.add_node("start", start)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("finish", finish)
    g.add_edge(START, "start")
    g.add_conditional_edges("start", after_start, ["agent", "finish"])
    g.add_conditional_edges("agent", after_agent, ["tools", "finish", END])
    g.add_conditional_edges("tools", after_tools, ["agent", "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def run_turn(ctx, llm, message: str, checkpointer=None, settings=None) -> dict:
    """Ejecuta un turno completo y devuelve la respuesta y la traza de tools."""
    graph = build_graph(ctx, llm, checkpointer, settings)
    config = {"configurable": {"thread_id": f"{ctx.plant_id}:{ctx.line_id}:{ctx.session_id}"},
              "recursion_limit": RECURSION_LIMIT}
    try:
        state = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    except Exception as e:  # GraphRecursionError u otro fallo del grafo: respuesta fija, hilo intacto
        audit(ctx, "graph", {}, "error", type(e).__name__)
        return {"reply": STOP_MESSAGES["max_tool_calls"], "tools_used": [], "stop_reason": "recursion",
                "pending_approval_id": None}
    last = state["messages"][-1]
    return {"reply": last.content if isinstance(last, AIMessage) else "", "tools_used": state.get("tools_used", []),
            "stop_reason": state.get("stop_reason"), "pending_approval_id": state.get("pending_approval_id")}
