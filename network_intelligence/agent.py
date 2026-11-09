"""Investigation agent loop.

The agent takes an incident, grounds itself in retrieved runbooks and
past incidents, then works the playbook step by step: it asks the
serving backend (planner) what to do next, calls the tool through the
hardened runner (retries / timeouts / dedup / context limits), and
finishes with a structured diagnostic report.

The planner is swappable: the mock backend plans by rule (a stand-in
for a fine-tuned diagnostic model), a real model would plan by
generating the next tool call. The loop itself does not care which.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from network_intelligence.config import AgentConfig
from network_intelligence.playbooks import select_playbook
from network_intelligence.tools import ToolRunner, make_tools

log = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    incident_id: str
    device: str
    root_cause: str
    confidence: float
    evidence: list[str]
    recommended_action: str
    playbook_id: str
    retrieved_docs: list[str]
    trace: list[dict]  # every tool call + result summary
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "device": self.device,
            "root_cause": self.root_cause,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "playbook_id": self.playbook_id,
            "retrieved_docs": self.retrieved_docs,
            "trace": self.trace,
            "summary": self.summary,
        }

    def render_text(self) -> str:
        lines = [
            f"Diagnostic report — {self.incident_id} ({self.device})",
            f"Root cause: {self.root_cause} (confidence {self.confidence:.2f})",
            f"Playbook: {self.playbook_id}",
            "",
            "Evidence:",
        ]
        lines += [f"  - {e}" for e in self.evidence] or ["  (none)"]
        lines += ["", f"Recommended action: {self.recommended_action}"]
        if self.retrieved_docs:
            lines += ["", "Grounded in: " + ", ".join(self.retrieved_docs)]
        lines += ["", f"Summary: {self.summary}"]
        return "\n".join(lines)


class InvestigationAgent:
    """Planner-driven investigation loop."""

    def __init__(
        self,
        backend,
        tools: dict,
        runner: ToolRunner | None = None,
        retriever=None,
        cfg: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.runner = runner or ToolRunner(cfg)
        self.retriever = retriever
        self.cfg = cfg or AgentConfig()

    def investigate(self, incident: dict) -> DiagnosticReport:
        device = incident.get("device", "")
        playbook = select_playbook(incident)

        # 1. Ground in runbooks + past incidents (RAG).
        retrieved: list[str] = []
        query = f"{incident.get('title', '')} {' '.join(incident.get('signals', {}))} {device}"
        if self.retriever is not None:
            try:
                hits = self.retriever.search(query, top_k=3)
                retrieved = [h.doc.id for h in hits]
            except Exception as exc:  # noqa: BLE001 - retrieval is advisory
                log.warning("retrieval failed: %s", exc)

        # 2. Work the plan step by step.
        history: list[dict] = []
        trace: list[dict] = []
        for _ in range(self.cfg.max_steps):
            action = self.backend.plan(incident, history, [{"id": d} for d in retrieved])
            if action.get("answer"):
                break
            tool_name = action.get("tool")
            args = dict(action.get("args", {}))
            args.setdefault("device", device)
            fn = self.tools.get(tool_name)
            if fn is None:
                trace.append({"tool": tool_name, "error": f"unknown tool {tool_name!r}"})
                continue
            result = self.runner.call(tool_name, fn, args)
            history.append({"tool": tool_name, "args": args, "result": result.to_dict()})
            trace.append({
                "tool": tool_name,
                "args": args,
                "cached": result.cached,
                "attempts": result.attempts,
                "error": result.error,
                "output_summary": _summarise(result.output),
            })
            if result.error:
                log.warning("tool %s failed after retries: %s", tool_name, result.error)

        # 3. Diagnose from the evidence.
        root_cause, confidence, evidence = self._diagnose(incident, history)
        summary = (
            f"{incident.get('title', incident.get('id', ''))} on {device}: "
            f"most likely {root_cause.replace('_', ' ')} "
            f"(confidence {confidence:.2f}). {playbook.remediation}"
        )
        return DiagnosticReport(
            incident_id=incident.get("id", "unknown"),
            device=device,
            root_cause=root_cause,
            confidence=confidence,
            evidence=evidence,
            recommended_action=playbook.remediation,
            playbook_id=playbook.id,
            retrieved_docs=retrieved,
            trace=trace,
            summary=summary,
        )

    def _diagnose(self, incident: dict, history: list[dict]) -> tuple[str, float, list[str]]:
        """Root cause + confidence + evidence from the tool history.

        The decision_tree tool is the source of truth for the branch;
        confidence grows with corroborating evidence (alerts, error
        logs, correlated events). If the tree says no-fault, the report
        says so — the agent must not invent a fault.
        """
        evidence: list[str] = []
        branch = "no-fault"
        for h in history:
            result = h["result"]
            evidence.extend(result.get("evidence", []))
            if h["tool"] == "decision_tree" and isinstance(result.get("output"), dict):
                branch = result["output"].get("branch", branch)

        confidence = 0.5
        if branch != "no-fault":
            confidence += 0.15
        if any("alert" in e.lower() for e in evidence):
            confidence += 0.1
        if any("ERROR" in e for e in evidence):
            confidence += 0.1
        if any("align" in e.lower() for e in evidence):
            confidence += 0.05
        confidence = min(confidence, 0.95)
        if branch == "no-fault":
            confidence = 0.6
        return branch, confidence, evidence


def _summarise(output: Any, limit: int = 300) -> str:
    text = json.dumps(output, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def build_agent(backend, metrics_store, incident_store, retriever=None, cfg: AgentConfig | None = None) -> InvestigationAgent:
    tools = make_tools(metrics_store, incident_store, retriever=retriever)
    return InvestigationAgent(backend, tools, ToolRunner(cfg), retriever, cfg or AgentConfig())
