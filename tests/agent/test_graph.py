import openai
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.budget import record_usage
from src.agent.graph import STOP_MESSAGES, run_turn
from src.common.config import Settings
from src.common.db import rows

from .conftest import IMG_ID


class ScriptedLLM:
    """Doble del LLM: devuelve respuestas guionizadas y cuenta las llamadas."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0) if self.script else AIMessage(content="fin")
        if isinstance(item, Exception):
            raise item
        return item


def call(name, args, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}],
                     usage_metadata={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050})


def test_happy_path_and_usage_recorded(make_ctx, db):
    llm = ScriptedLLM([call("analyze_image", {"image_id": IMG_ID}),
                       AIMessage(content="Producto 7 a 120°", usage_metadata={"input_tokens": 1200, "output_tokens": 30,
                                                                             "total_tokens": 1230})])
    out = run_turn(make_ctx(), llm, "¿qué es esta imagen?", InMemorySaver())
    assert out["reply"] == "Producto 7 a 120°" and out["tools_used"][0]["tool"] == "analyze_image"
    assert rows(db, "SELECT COUNT(*) AS n FROM llm_usage")[0]["n"] == 2


def test_loop_cut_on_third_identical_call(make_ctx):
    llm = ScriptedLLM([call("get_camera_health", {}, i) for i in range(6)])
    out = run_turn(make_ctx(), llm, "repite", InMemorySaver())
    assert out["stop_reason"] == "loop_detected" and out["reply"] == STOP_MESSAGES["loop_detected"]
    assert [t["status"] for t in out["tools_used"]] == ["ok", "ok", "loop_detected"]


def test_risk_request_pending_not_executed(make_ctx, db):
    llm = ScriptedLLM([call("request_risk_operation", {"op": "pause_line", "params": {},
                                                       "justification": "limpieza de lente"}),
                       AIMessage(content="He pedido la pausa; queda pendiente de aprobación.")])
    out = run_turn(make_ctx(), llm, "pausa la línea", InMemorySaver())
    assert out["pending_approval_id"]
    assert rows(db, "SELECT paused FROM line_state WHERE plant_id='P1'")[0]["paused"] == 0


def test_budget_exhausted_no_llm_call(make_ctx, db):
    record_usage(db, "s0", "x", {"input_tokens": 10_000_000, "output_tokens": 0})
    llm = ScriptedLLM([AIMessage(content="no debería")])
    out = run_turn(make_ctx(), llm, "hola", InMemorySaver())
    assert llm.calls == 0 and out["reply"] == STOP_MESSAGES["budget"]


def test_llm_down_fixed_message_thread_intact(make_ctx):
    saver = InMemorySaver()
    err = openai.APIConnectionError(request=__import__("httpx").Request("POST", "http://x"))
    out = run_turn(make_ctx(), ScriptedLLM([err]), "hola", saver)
    assert out["reply"] == STOP_MESSAGES["llm_unavailable"]
    out2 = run_turn(make_ctx(), ScriptedLLM([AIMessage(content="ya estoy")]), "¿sigues?", saver)
    assert out2["reply"] == "ya estoy"


def test_invalid_args_returned_to_model(make_ctx):
    llm = ScriptedLLM([call("analyze_image", {"image_id": IMG_ID, "top_k": 50}),
                       AIMessage(content="Argumento corregido")])
    out = run_turn(make_ctx(), llm, "x", InMemorySaver())
    assert out["tools_used"][0]["status"] == "invalid_args" and out["reply"] == "Argumento corregido"


def test_history_trimmed_with_summary(make_ctx):
    saver = InMemorySaver()
    for i in range(8):
        run_turn(make_ctx(), ScriptedLLM([call("get_camera_health", {"start_hour": i}),
                                          AIMessage(content=f"r{i}")]), f"pregunta {i}", saver)
    from src.agent.graph import build_graph
    g = build_graph(make_ctx(), ScriptedLLM([]), saver)
    st = g.get_state({"configurable": {"thread_id": "P1:L1:s1"}}).values
    assert len(st["messages"]) <= 12 and "pregunta 0" in st["summary"]


@pytest.mark.parametrize("fault", ["500", "timeout"])
def test_vision_failure_reported(make_ctx, fault):
    ctx = make_ctx()
    make_ctx.vision.fault = fault
    llm = ScriptedLLM([call("analyze_image", {"image_id": IMG_ID}), AIMessage(content="No pude analizarla")])
    out = run_turn(ctx, llm, "analiza", InMemorySaver())
    assert out["tools_used"][0]["status"] == f"vision_{'server' if fault == '500' else 'timeout'}"


def test_settings_caps_respected(make_ctx, db):
    s = Settings(daily_usd_cap=0.0)
    out = run_turn(make_ctx(), ScriptedLLM([AIMessage(content="x")]), "hola", InMemorySaver(), settings=s)
    assert out["reply"] == STOP_MESSAGES["budget"]
