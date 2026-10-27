"""Tope de gasto en código: la clave de OpenAI no tiene límite propio, así que lo aplica la aplicación."""
from datetime import datetime, timezone

from src.common.config import get_settings
from src.common.db import now_iso


def cost_usd(usage: dict, settings=None) -> float:
    s = settings or get_settings()
    inp = int(usage.get("input_tokens", 0))
    cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
    out = int(usage.get("output_tokens", 0))
    return ((inp - cached) * s.price_input_per_m + cached * s.price_cached_per_m + out * s.price_output_per_m) / 1e6


def record_usage(conn, session_id: str, model: str, usage: dict, settings=None) -> float:
    usd = cost_usd(usage, settings)
    cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
    conn.execute("INSERT INTO llm_usage (ts, session_id, model, input_tokens, cached_tokens, output_tokens, usd)"
                 " VALUES (?,?,?,?,?,?,?)",
                 (now_iso(), session_id, model, int(usage.get("input_tokens", 0)), cached,
                  int(usage.get("output_tokens", 0)), usd))
    conn.commit()
    return usd


def spent(conn) -> tuple[float, float]:
    today = datetime.now(timezone.utc).date().isoformat()
    total = conn.execute("SELECT COALESCE(SUM(usd),0) FROM llm_usage").fetchone()[0]
    day = conn.execute("SELECT COALESCE(SUM(usd),0) FROM llm_usage WHERE substr(ts,1,10)=?", (today,)).fetchone()[0]
    return float(day), float(total)


def budget_ok(conn, settings=None) -> bool:
    s = settings or get_settings()
    day, total = spent(conn)
    return day < s.daily_usd_cap and total < s.total_usd_cap
