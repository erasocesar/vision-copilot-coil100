from src.agent.audit import args_hash
from src.agent.guard import MAX_TOOL_CALLS, check_call

from .conftest import IMG_ID


def test_unknown_tool(make_ctx):
    r = check_call(make_ctx(), "drop_tables", {}, [])
    assert not r.ok and r.error["error"] == "unknown_tool" and not r.stop


def test_invalid_args(make_ctx):
    r = check_call(make_ctx(), "analyze_image", {"image_id": "zz"}, [])
    assert r.error["error"] == "invalid_args"


def test_forbidden(make_ctx):
    r = check_call(make_ctx(role="maintenance"), "submit_review_label", {"case_id": "x", "obj_id": 1, "angle": 1}, [])
    assert r.error["error"] == "forbidden"


def test_loop_third_identical_call(make_ctx):
    args = {"image_id": IMG_ID}
    calls = [("analyze_image", args_hash(args))] * 2
    r = check_call(make_ctx(), "analyze_image", args, calls)
    assert r.error["error"] == "loop_detected" and r.stop
    assert check_call(make_ctx(), "analyze_image", args, calls[:1]).ok


def test_max_calls(make_ctx):
    calls = [("query_line_metrics", str(i)) for i in range(MAX_TOOL_CALLS)]
    r = check_call(make_ctx(), "get_camera_health", {}, calls)
    assert r.error["error"] == "max_tool_calls" and r.stop


def test_ok(make_ctx):
    assert check_call(make_ctx(), "get_camera_health", {}, []).ok
