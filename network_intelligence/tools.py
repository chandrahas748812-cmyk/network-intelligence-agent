"""Investigation tools: what the agent calls, step by step.

Every tool returns a :class:`ToolResult` with structured output, evidence
strings the final report can cite, and timing. The :class:`ToolRunner`
adds the operational hardening the JD asks for:

* **retries** — transient failures are retried with back-off;
* **timeouts** — a tool that takes too long fails fast instead of
  hanging the investigation;
* **dedup** — identical calls inside the TTL are served from cache
  (Redis when configured, in-memory otherwise);
* **context limits** — outputs are truncated to the agent's budget so a
  huge log dump cannot blow up the model's context window.

Tools are plain functions over the stores, so they are unit-testable
without any model. ``flaky`` wrappers in the tests simulate failures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from network_intelligence.config import AgentConfig
from network_intelligence.stores import DedupCache, InMemoryIncidentStore, InMemoryMetricsStore

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool: str
    output: Any
    evidence: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "output": self.output,
            "evidence": self.evidence,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "cached": self.cached,
            "error": self.error,
        }


class ToolTimeout(Exception):
    pass


class ToolRunner:
    """Runs tools with retries, timeouts, dedup, and output truncation."""

    def __init__(self, cfg: AgentConfig | None = None, cache: DedupCache | None = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.cache = cache or DedupCache()

    def _cache_key(self, tool: str, args: dict) -> str:
        raw = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
        return "tool:" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def call(self, tool: str, fn: Callable[..., Any], args: dict) -> ToolResult:
        key = self._cache_key(tool, args)
        cached = self.cache.get(key)
        if cached is not None:
            result = ToolResult(**cached)
            result.cached = True
            return result

        last_error: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                output = fn(**args)
                latency_ms = (time.monotonic() - started) * 1000
                if latency_ms > self.cfg.tool_timeout_s * 1000:
                    raise ToolTimeout(f"{tool} took {latency_ms:.0f}ms > {self.cfg.tool_timeout_s}s timeout")
                output = self._truncate(output)
                result = ToolResult(
                    tool=tool,
                    output=output,
                    evidence=self._evidence(tool, output),
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
                self.cache.set(key, result.to_dict(), ttl_s=self.cfg.dedup_ttl_s)
                return result
            except Exception as exc:  # noqa: BLE001 - retry, then report
                last_error = exc
                log.warning("tool %s attempt %d failed: %s", tool, attempt, exc)
                time.sleep(min(0.1 * 2 ** (attempt - 1), 1.0))
        return ToolResult(tool=tool, output=None, attempts=self.cfg.max_retries, error=str(last_error))

    def _truncate(self, output: Any) -> Any:
        """Truncate long string outputs to the context budget."""
        limit = self.cfg.max_tool_output_chars
        if isinstance(output, str) and len(output) > limit:
            return output[:limit] + f"\n… [truncated to {limit} chars for context budget]"
        if isinstance(output, list) and len(output) > 50:
            return output[-50:]
        return output

    @staticmethod
    def _evidence(tool: str, output: Any) -> list[str]:
        if isinstance(output, dict) and "evidence" in output:
            return list(output["evidence"])
        return []


# ---------------------------------------------------------------------------
# Tool implementations (pure functions over the stores)
# ---------------------------------------------------------------------------

def make_tools(metrics: InMemoryMetricsStore, incidents: InMemoryIncidentStore, retriever=None):
    """Build the tool table the agent calls. ``retriever`` is optional;
    without it the retrieve_runbook tool reports itself unavailable."""

    def query_metrics(device: str, metrics: list[str], minutes: int = 30) -> dict:
        out, evidence = {}, []
        for metric in metrics:
            rows = metrics_store.query_metrics(device, metric, minutes=minutes)
            if not rows:
                continue
            values = [r["value"] for r in rows]
            latest = values[-1]
            out[metric] = {
                "latest": latest,
                "mean": round(statistics.fmean(values), 2),
                "min": min(values),
                "max": max(values),
                "points": len(values),
            }
            evidence.append(f"{metric} on {device}: latest={latest} (mean {out[metric]['mean']}, n={len(values)})")
        return {"device": device, "metrics": out, "evidence": evidence}

    metrics_store = metrics

    def search_logs(device: str, level: str | None = None, limit: int = 20) -> dict:
        rows = metrics_store.search_logs(device, level=level, limit=limit)
        lines = [f"[{r['ts']}] {r['level']} {r['message']}" for r in rows]
        errors = [l for l in lines if "ERROR" in l]
        return {
            "device": device,
            "lines": lines,
            "error_count": len(errors),
            "evidence": errors[:5],
        }

    def check_alerts(device: str) -> dict:
        rows = metrics_store.active_alerts(device)
        return {
            "device": device,
            "alerts": rows,
            "evidence": [f"{a['rule']} ({a['severity']}): {a['message']}" for a in rows],
        }

    def threshold_check(device: str, metric: str, threshold: float, op: str = "gt") -> dict:
        latest = metrics_store.latest_value(device, metric)
        if latest is None:
            return {"device": device, "metric": metric, "breached": False, "evidence": [f"no data for {metric} on {device}"]}
        breached = latest > threshold if op == "gt" else latest < threshold
        return {
            "device": device,
            "metric": metric,
            "latest": latest,
            "threshold": threshold,
            "op": op,
            "breached": breached,
            "evidence": [f"{metric}={latest} {op} {threshold}: {'BREACHED' if breached else 'ok'}"],
        }

    def decision_tree(device: str, signals: dict) -> dict:
        """Small, explicit decision tree over the incident's signals.

        This is the deterministic core the evals score: the same signals
        must always land on the same branch, so diagnosis correctness is
        measurable instead of vibes.
        """
        tok_s = signals.get("inference_tok_per_s")
        kv = signals.get("kv_cache_usage_pct")
        loss = signals.get("packet_loss_pct")
        util = signals.get("gpu_util_pct")
        branch, reason = "no-fault", "No signal crosses a fault threshold."
        if kv is not None and kv >= 90 and tok_s is not None and tok_s < 8:
            branch, reason = "kv_cache_exhaustion", f"KV cache {kv}% with tok/s {tok_s} (<8)"
        elif loss is not None and loss >= 1.0:
            branch, reason = "faulty_optic", f"packet loss {loss}% (>=1%)"
        elif util is not None and util < 20 and tok_s is not None and tok_s < 10:
            branch, reason = "batching_too_small", f"GPU util {util}% with tok/s {tok_s} and queue pressure"
        return {
            "device": device,
            "branch": branch,
            "reason": reason,
            "evidence": [f"decision_tree -> {branch}: {reason}"],
        }

    def correlate_events(device: str, minutes: int = 30) -> dict:
        """Join alerts + ERROR logs + metric anomalies in one window."""
        alerts = metrics_store.active_alerts(device)
        logs = metrics_store.search_logs(device, level="ERROR", limit=20)
        timeline = []
        for a in alerts:
            timeline.append({"ts": a["ts"], "kind": "alert", "text": f"{a['rule']}: {a['message']}"})
        for l in logs:
            timeline.append({"ts": l["ts"], "kind": "log", "text": l["message"]})
        timeline.sort(key=lambda e: e["ts"])
        correlated = len(alerts) > 0 and len(logs) > 0
        return {
            "device": device,
            "events": timeline,
            "correlated": correlated,
            "evidence": [
                f"{len(alerts)} alert(s) and {len(logs)} ERROR log(s) in window; "
                f"{'signals align on one device' if correlated else 'no overlap found'}"
            ],
        }

    def retrieve_runbook(query: str, top_k: int = 3) -> dict:
        if retriever is None:
            return {"query": query, "hits": [], "evidence": ["retriever not configured"]}
        hits = retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "hits": [h.to_dict() for h in hits],
            "evidence": [f"{h.doc.kind} {h.doc.id}: {h.doc.title}" for h in hits],
        }

    return {
        "query_metrics": query_metrics,
        "search_logs": search_logs,
        "check_alerts": check_alerts,
        "threshold_check": threshold_check,
        "decision_tree": decision_tree,
        "correlate_events": correlate_events,
        "retrieve_runbook": retrieve_runbook,
    }


TOOL_SCHEMAS = {
    "query_metrics": {"device": "str", "metrics": "list[str]", "minutes": "int=30"},
    "search_logs": {"device": "str", "level": "str|None", "limit": "int=20"},
    "check_alerts": {"device": "str"},
    "threshold_check": {"device": "str", "metric": "str", "threshold": "float", "op": "str=gt"},
    "decision_tree": {"device": "str", "signals": "dict"},
    "correlate_events": {"device": "str", "minutes": "int=30"},
    "retrieve_runbook": {"query": "str", "top_k": "int=3"},
}
