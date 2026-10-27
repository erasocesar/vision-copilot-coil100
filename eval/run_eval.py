"""Evaluación reproducible del agente: casos en YAML, criterio objetivo por caso, tokens y coste.

Uso:
  python eval/run_eval.py --cases eval/cases.yaml --repeats 1 --out eval/results/run.json
  python eval/run_eval.py --only C01,C02          # subconjunto
El servicio de visión corre en el mismo proceso (TestClient) con el modelo v1.0.0; cada caso parte de
una copia limpia de la base simulada. Necesita OPENAI_API_KEY en .env.
"""
import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import approvals  # noqa: E402
from src.agent.api import make_llm  # noqa: E402
from src.agent.context import ToolContext  # noqa: E402
from src.agent.graph import run_turn  # noqa: E402
from src.agent.vision_client import VisionClient  # noqa: E402
from src.common.config import Settings  # noqa: E402
from src.common.db import connect, rows  # noqa: E402
from src.vision.api import create_app as create_vision  # noqa: E402

WORK = Path("eval/.work")
EST_USD_PER_CASE = 0.02


def build_template(settings: Settings) -> Path:
    """Base simulada con el modelo real (una vez); cada caso trabaja sobre una copia."""
    tpl = WORK / "template.db"
    if not tpl.exists():
        import subprocess
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "src.sim.line_sim", "--db", str(tpl), "--model", str(settings.model_dir),
                        "--image-dir", str(settings.image_dir)], check=True)
    return tpl


def upload(vision: VisionClient, path: Path) -> str:
    r = vision.http.post("/v1/quality", files={"file": (path.name, path.read_bytes(), "image/png")})
    r.raise_for_status()
    return r.json()["image_id"]


def check_args(expected: dict, used: list) -> bool:
    for tool, fields in (expected or {}).items():
        calls = [u["args"] for u in used if u["tool"] == tool]
        if not calls:
            return False
        ok_any = False
        for args in calls:
            ok = True
            for k, v in fields.items():
                got = args.get(k)
                if isinstance(v, str) and v.startswith("re:"):
                    ok &= got is not None and re.search(v[3:], str(got)) is not None
                else:
                    ok &= got == v
            ok_any |= ok
        if not ok_any:
            return False
    return True


def check_success(spec: list, out: dict, conn, ctx) -> tuple[bool, list]:
    failed = []
    for s in spec or []:
        t = s["type"]
        if t == "reply_regex":
            ok = re.search(s["pattern"], out["reply"], re.IGNORECASE | re.DOTALL) is not None
        elif t == "reply_not_regex":
            ok = re.search(s["pattern"], out["reply"], re.IGNORECASE | re.DOTALL) is None
        elif t == "db":
            val = rows(conn, s["sql"], (ctx.plant_id, ctx.line_id))[0]
            ok = list(val.values())[0] == s["expect"]
        elif t == "stop_reason":
            ok = out["stop_reason"] == s["value"]
        elif t == "pending_approval":
            ok = bool(out["pending_approval_id"]) == s["value"]
        elif t == "max_tool_calls":
            ok = sum(u["tool"] == s["tool"] for u in out["tools_used"]) <= s["max"]
        elif t == "tool_status":
            ok = any(u["tool"] == s["tool"] and u["status"] == s["status"] for u in out["tools_used"])
        else:
            raise ValueError(t)
        if not ok:
            failed.append(s)
    return not failed, failed


def run_case(case: dict, settings: Settings, llm, vision: VisionClient, tpl: Path, assets: Path) -> dict:
    db = WORK / f"case_{case['id']}.db"
    shutil.copy(tpl, db)
    conn = connect(db)
    sid = f"eval-{case['id']}-{int(time.time() * 1000)}"
    ctx = ToolContext(case.get("user", "eval_user"), case["role"], case.get("plant", "P1"), "L1", sid, conn,
                      vision, str(settings.image_dir))
    for setup in case.get("setup", []):
        if setup.get("sql"):
            conn.execute(setup["sql"])
            conn.commit()
        if setup.get("request"):
            r = setup["request"]
            req_ctx = ToolContext(r["user"], r["role"], "P1", "L1", "seed", conn, vision, str(settings.image_dir))
            approvals.create_request(req_ctx, r["op"], r.get("params", {}), r.get("justification", "sembrado"))
    ids = [upload(vision, assets / p) for p in case.get("images", [])]
    sku = case.get("sku_views")
    msg = case["message"]
    if ids:
        msg += "\n[imágenes adjuntas: " + ", ".join(ids) + "]"
    if sku:
        files = sorted((assets / sku["dir"]).glob("view_*.png"))
        files = [f for f in files if int(f.stem.split("_")[1]) % sku["every"] == 0]
        vids = [upload(vision, f) for f in files]
        angles = [int(f.stem.split("_")[1]) for f in files]
        msg += f"\n[vistas del producto nuevo: image_ids={vids} angles={angles}]"
    vision.failures, vision.open_until = 0, 0.0
    fault = case.get("vision_fault")
    original_post = vision.http.post
    if fault == "timeout":
        # En proceso (TestClient) no hay timeout de red: se simula la excepción del cliente HTTP.
        def post_timeout(url, **kw):
            raise httpx.ReadTimeout("timeout simulado", request=httpx.Request("POST", "http://vision" + url))
        vision.http.post = post_timeout
    else:
        vision.fault = fault
    t0 = time.perf_counter()
    try:
        out = run_turn(ctx, llm, msg, InMemorySaver(), settings)
    finally:
        vision.http.post = original_post
        vision.fault = None
    latency = time.perf_counter() - t0
    usage = rows(conn, "SELECT COALESCE(SUM(input_tokens),0) AS i, COALESCE(SUM(cached_tokens),0) AS c,"
                       " COALESCE(SUM(output_tokens),0) AS o, COALESCE(SUM(usd),0) AS usd, COUNT(*) AS calls"
                       " FROM llm_usage WHERE session_id=?", (sid,))[0]
    used_tools = [u["tool"] for u in out["tools_used"]]
    expected = case.get("expected_tools", [])
    tools_ok = all(t in used_tools for t in expected) and not any(t in used_tools for t in case.get("forbidden_tools", []))
    args_ok = check_args(case.get("arg_checks"), out["tools_used"])
    success, failed = check_success(case.get("success"), out, conn, ctx)
    conn.close()
    return {"id": case["id"], "use_case": case["use_case"], "adverse": case.get("adverse", False),
            "task_success": success, "tool_selection_ok": tools_ok, "args_ok": args_ok,
            "tools_used": out["tools_used"], "reply": out["reply"], "stop_reason": out["stop_reason"],
            "failed_checks": failed, "llm_calls": usage["calls"], "input_tokens": usage["i"],
            "cached_tokens": usage["c"], "output_tokens": usage["o"], "usd": round(usage["usd"], 5),
            "latency_s": round(latency, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("eval/cases.yaml"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-usd", type=float, default=1.5, help="no empieza si el coste estimado lo supera")
    args = ap.parse_args()

    settings = Settings(image_dir=WORK / "images", db_path=WORK / "unused.db", demo_faults=True,
                        daily_usd_cap=1e9, total_usd_cap=1e9)
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    est = len(cases) * args.repeats * EST_USD_PER_CASE
    if est > args.max_usd:
        sys.exit(f"coste estimado {est:.2f} USD supera --max-usd {args.max_usd}")
    llm = make_llm(settings)
    if llm is None:
        sys.exit("falta OPENAI_API_KEY en .env")
    vision_app = create_vision(Settings(model_dir=settings.model_dir, image_dir=settings.image_dir, demo_faults=True))
    vision = VisionClient("http://vision", sleep=lambda s: None)
    vision.http = TestClient(vision_app, raise_server_exceptions=False)
    tpl = build_template(settings)
    assets = Path("eval/assets")

    results = []
    for rep in range(args.repeats):
        for case in cases:
            r = run_case(case, settings, llm, vision, tpl, assets)
            r["repeat"] = rep
            results.append(r)
            print(f"{r['id']:>4} rep{rep} {'OK ' if r['task_success'] else 'FAIL'} tools={'ok' if r['tool_selection_ok'] else 'X'}"
                  f" args={'ok' if r['args_ok'] else 'X'} {r['usd']:.4f} USD  {[u['tool'] for u in r['tools_used']]}",
                  flush=True)

    n = len(results)
    summary = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": settings.openai_model,
        "cases": len(cases), "repeats": args.repeats, "runs": n,
        "task_success": round(sum(r["task_success"] for r in results) / n, 3),
        "tool_selection": round(sum(r["tool_selection_ok"] for r in results) / n, 3),
        "args": round(sum(r["args_ok"] for r in results) / n, 3),
        "usd_total": round(sum(r["usd"] for r in results), 4), "usd_per_run": round(sum(r["usd"] for r in results) / n, 5),
        "tokens_in_per_run": round(sum(r["input_tokens"] for r in results) / n),
        "tokens_out_per_run": round(sum(r["output_tokens"] for r in results) / n),
    }
    print(json.dumps(summary, indent=2))
    out = args.out or Path(f"eval/results/agent_{datetime.now():%Y%m%d_%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("guardado en", out)


if __name__ == "__main__":
    main()
