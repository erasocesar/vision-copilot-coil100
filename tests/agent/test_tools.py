from src.agent.tools import TOOLS, openai_tool_specs, run_tool
from src.common.db import rows

from .conftest import IMG_ID

EXPECTED = {"check_image_quality", "analyze_image", "list_review_queue", "submit_review_label",
            "query_line_metrics", "get_camera_health", "get_grasp_events", "search_procedures", "create_ticket",
            "register_sku_candidate", "evaluate_sku_candidate", "request_risk_operation", "generate_shift_report"}


def test_catalog_has_13_tools_read_and_write():
    assert set(TOOLS) == EXPECTED
    assert sum(t.side_effect for t in TOOLS.values()) >= 4
    assert len(openai_tool_specs()) == 13


def test_analyze_image(make_ctx):
    out = run_tool(make_ctx(), "analyze_image", {"image_id": IMG_ID, "top_k": 2})
    assert out["obj_id"] == 7 and out["action"] == "PICK" and len(out["evidence_topk"]) == 2
    assert out["image_id"] == IMG_ID
    assert "path" not in out["evidence_topk"][0]


def test_analyze_image_uses_line_threshold(make_ctx, db):
    db.execute("UPDATE line_state SET min_pick_score=0.999 WHERE plant_id='P1'")
    out = run_tool(make_ctx(), "analyze_image", {"image_id": IMG_ID})
    assert out["action"] == "DIVERT" and "umbral de línea" in out["reasons"][0]


def test_analyze_unknown_image(make_ctx):
    out = run_tool(make_ctx(), "analyze_image", {"image_id": "b" * 64})
    assert out["error"] == "image_not_found"


def test_invalid_args(make_ctx):
    out = run_tool(make_ctx(), "analyze_image", {"image_id": IMG_ID, "top_k": 50})
    assert out["error"] == "invalid_args"


def test_unknown_tool(make_ctx):
    assert run_tool(make_ctx(), "borrar_todo", {})["error"] == "unknown_tool"


def test_vision_failure_is_structured(make_ctx):
    ctx = make_ctx()
    make_ctx.vision.fault = "500"
    out = run_tool(ctx, "analyze_image", {"image_id": IMG_ID})
    assert out["error"] == "vision_server" and "no inventes" in out["hint"]


def test_forbidden_role_audited(make_ctx, db):
    out = run_tool(make_ctx(role="maintenance"), "submit_review_label", {"case_id": "x", "obj_id": 1, "angle": 0})
    assert out["error"] == "forbidden"
    assert rows(db, "SELECT status FROM audit ORDER BY id DESC LIMIT 1")[0]["status"] == "forbidden"


def test_every_call_audited(make_ctx, db):
    n0 = rows(db, "SELECT COUNT(*) AS n FROM audit")[0]["n"]
    run_tool(make_ctx(), "query_line_metrics", {"group_by": "shift"})
    run_tool(make_ctx(), "get_camera_health", {})
    assert rows(db, "SELECT COUNT(*) AS n FROM audit")[0]["n"] == n0 + 2


def test_metrics_isolated_by_plant(make_ctx):
    p1 = run_tool(make_ctx(plant="P1"), "query_line_metrics", {"group_by": "shift"})
    p2 = run_tool(make_ctx(plant="P2"), "query_line_metrics", {"group_by": "shift"})
    assert p1["total"] == 1440 and p2["total"] < p1["total"]


def test_metrics_3am(make_ctx):
    out = run_tool(make_ctx(), "query_line_metrics", {"start_hour": 2, "end_hour": 5, "group_by": "hour"})
    by_hour = {r["bucket"]: r["divert_rate"] for r in out["rows"]}
    assert by_hour["03"] > by_hour["04"]


def test_metrics_shift_and_hours_conflict(make_ctx):
    assert run_tool(make_ctx(), "query_line_metrics", {"shift": 2, "start_hour": 3})["error"] == "invalid_args"


def test_review_queue_other_plant_case_not_found(make_ctx):
    p2_cases = run_tool(make_ctx(plant="P2"), "list_review_queue", {"limit": 20})["cases"]
    assert p2_cases
    out = run_tool(make_ctx(plant="P1"), "submit_review_label",
                   {"case_id": p2_cases[0]["case_id"], "obj_id": 1, "angle": 0})
    assert out["error"] == "not_found"


def test_label_idempotent(make_ctx, db):
    case = run_tool(make_ctx(), "list_review_queue", {"limit": 1})["cases"][0]["case_id"]
    a = run_tool(make_ctx(), "submit_review_label", {"case_id": case, "obj_id": 3, "angle": 40})
    b = run_tool(make_ctx(), "submit_review_label", {"case_id": case, "obj_id": 3, "angle": 40})
    c = run_tool(make_ctx(), "submit_review_label", {"case_id": case, "obj_id": 4, "angle": 40})
    assert a["created"] and not b["created"] and c["error"] == "conflict"
    assert rows(db, "SELECT COUNT(*) AS n FROM retrain_dataset")[0]["n"] == 1


def test_grasp_events_vacuum(make_ctx):
    out = run_tool(make_ctx(role="maintenance"), "get_grasp_events", {"failure_code": "VACUUM_LOSS"})
    assert out["failures_by_code_hour"] and all(r["hour"] == "17" for r in out["failures_by_code_hour"])


def test_search_procedures_with_citation(make_ctx):
    out = run_tool(make_ctx(role="maintenance"), "search_procedures", {"query": "pérdida de vacío ventosa agarre"})
    assert out["matches"][0]["citation"].startswith("SOP-ROB-02#")


def test_search_procedures_no_match(make_ctx):
    out = run_tool(make_ctx(), "search_procedures", {"query": "receta de paella valenciana"})
    assert out["matches"] == []


def test_ticket_idempotent(make_ctx):
    args = {"title": "Pérdida de vacío en ventosa", "priority": "high",
            "body": "Fallos VACUUM_LOSS entre 17 y 18 h. Ver SOP-ROB-02.", "related_inspection_ids": [3, 1]}
    a = run_tool(make_ctx(role="maintenance"), "create_ticket", args)
    b = run_tool(make_ctx(role="maintenance"), "create_ticket", {**args, "related_inspection_ids": [1, 3]})
    assert a["created"] and not b["created"] and a["ticket_id"] == b["ticket_id"]


def test_sku_register_insufficient(make_ctx):
    ids = [f"{i:064x}" for i in range(12)]
    out = run_tool(make_ctx(role="ml_engineer"), "register_sku_candidate",
                   {"name": "Nuevo envase", "image_ids": ids, "angles": list(range(0, 360, 30))})
    assert out["status"] == "insufficient" and out["max_gap_deg"] == 30
    ev = run_tool(make_ctx(role="ml_engineer"), "evaluate_sku_candidate", {"candidate_id": out["candidate_id"]})
    assert ev["error"] == "insufficient_views"


def test_sku_register_and_evaluate(make_ctx):
    ids = [f"{i:064x}" for i in range(12)] * 3
    angles = list(range(0, 360, 10))
    out = run_tool(make_ctx(role="ml_engineer"), "register_sku_candidate",
                   {"name": "Nuevo envase", "image_ids": ids, "angles": angles})
    assert out["status"] == "registered"
    ev = run_tool(make_ctx(role="ml_engineer"), "evaluate_sku_candidate", {"candidate_id": out["candidate_id"]})
    assert ev["status"] == "evaluated_ok"


def test_sku_evaluate_cannot_overwrite_production(make_ctx, db):
    ids = [f"{i:064x}" for i in range(12)] * 3
    out = run_tool(make_ctx(role="ml_engineer"), "register_sku_candidate",
                   {"name": "Nuevo envase", "image_ids": ids, "angles": list(range(0, 360, 10))})
    cid = out["candidate_id"]
    run_tool(make_ctx(role="ml_engineer"), "evaluate_sku_candidate", {"candidate_id": cid})
    db.execute("UPDATE sku_candidates SET status='production' WHERE candidate_id=?", (cid,))
    db.commit()
    metrics_before = rows(db, "SELECT metrics_json FROM sku_candidates WHERE candidate_id=?", (cid,))[0]
    ev = run_tool(make_ctx(role="quality"), "evaluate_sku_candidate", {"candidate_id": cid})
    assert ev["error"] == "invalid_state"
    after = rows(db, "SELECT status, metrics_json FROM sku_candidates WHERE candidate_id=?", (cid,))[0]
    assert after["status"] == "production" and after["metrics_json"] == metrics_before["metrics_json"]


def test_sku_evaluate_does_not_overwrite_promotion_during_evaluation(make_ctx, db):
    ids = [f"{i:064x}" for i in range(12)] * 3
    out = run_tool(make_ctx(role="ml_engineer"), "register_sku_candidate",
                   {"name": "Nuevo envase", "image_ids": ids, "angles": list(range(0, 360, 10))})
    cid = out["candidate_id"]
    ctx = make_ctx(role="ml_engineer")
    original = ctx.vision.evaluate_views

    def promote_while_evaluating(paths, angles):  # simula una aprobación que llega durante la evaluación
        db.execute("UPDATE sku_candidates SET status='production' WHERE candidate_id=?", (cid,))
        db.commit()
        return original(paths, angles)
    ctx.vision.evaluate_views = promote_while_evaluating
    ev = run_tool(ctx, "evaluate_sku_candidate", {"candidate_id": cid})
    assert ev["error"] == "invalid_state"
    assert rows(db, "SELECT status FROM sku_candidates WHERE candidate_id=?", (cid,))[0]["status"] == "production"


def test_risk_operation_never_executes(make_ctx, db):
    out = run_tool(make_ctx(role="quality"), "request_risk_operation",
                   {"op": "set_threshold", "params": {"min_pick_score": 0.9}, "justification": "demasiados desvíos"})
    assert out["status"] == "pending_approval"
    assert rows(db, "SELECT min_pick_score FROM line_state WHERE plant_id='P1'")[0]["min_pick_score"] == 0.97


def test_risk_operation_bad_params(make_ctx):
    out = run_tool(make_ctx(role="quality"), "request_risk_operation",
                   {"op": "set_threshold", "params": {"min_pick_score": 1.2}, "justification": "fuera de rango"})
    assert out["error"] == "invalid_request"


def test_shift_report(make_ctx):
    out = run_tool(make_ctx(), "generate_shift_report", {"shift": 3})
    assert out["total"] > 0 and any("03:00" in i for i in out["incidents"])
    assert "pending_review_cases" in out and out["line_state"]["model_version"] == "1.0.0"


def test_sku_rejects_path_like_ids(make_ctx):
    ids = ["../../etc/passwd"] + [f"{i:064x}" for i in range(5)]
    out = run_tool(make_ctx(role="ml_engineer"), "register_sku_candidate",
                   {"name": "x envase", "image_ids": ids, "angles": list(range(0, 60, 10))})
    assert out["error"] == "invalid_args"

