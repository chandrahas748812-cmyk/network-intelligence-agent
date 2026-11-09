"""Eval harness: does the diagnosis actually come out right?

The JD is explicit — evals must measure *whether the diagnosis is
actually correct, not just whether the loss went down*. So this harness
scores the agent against golden incident cases on three things:

* **root-cause accuracy** — the agent's root cause matches the expected
  one (exact string match on the canonical cause id);
* **evidence quality** — the fraction of the case's expected evidence
  keywords that appear in the report's evidence;
* **false-alarm rate** — on healthy cases (expected root cause
  ``"no-fault"``), how often the agent claims a fault anyway.

Run it with ``python -m network_intelligence.evals``. The demo prints
the same summary. Golden cases live in ``data/golden_cases.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CaseResult:
    case_id: str
    expected: str
    predicted: str
    correct: bool
    evidence_score: float
    false_alarm: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "expected": self.expected,
            "predicted": self.predicted,
            "correct": self.correct,
            "evidence_score": round(self.evidence_score, 2),
            "false_alarm": self.false_alarm,
        }


@dataclass
class EvalResult:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def root_cause_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.correct for c in self.cases) / len(self.cases)

    @property
    def mean_evidence_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.evidence_score for c in self.cases) / len(self.cases)

    @property
    def false_alarm_rate(self) -> float:
        healthy = [c for c in self.cases if c.expected == "no-fault"]
        if not healthy:
            return 0.0
        return sum(c.false_alarm for c in healthy) / len(healthy)

    def to_dict(self) -> dict:
        return {
            "n_cases": len(self.cases),
            "root_cause_accuracy": round(self.root_cause_accuracy, 3),
            "mean_evidence_score": round(self.mean_evidence_score, 3),
            "false_alarm_rate": round(self.false_alarm_rate, 3),
            "cases": [c.to_dict() for c in self.cases],
        }

    def render_text(self) -> str:
        lines = [
            f"Eval: {len(self.cases)} golden cases",
            f"  root-cause accuracy: {self.root_cause_accuracy:.0%}",
            f"  mean evidence score: {self.mean_evidence_score:.2f}",
            f"  false-alarm rate:    {self.false_alarm_rate:.0%}",
        ]
        for c in self.cases:
            mark = "OK " if c.correct else "MISS"
            lines.append(f"  [{mark}] {c.case_id}: expected {c.expected}, got {c.predicted} (evidence {c.evidence_score:.2f})")
        return "\n".join(lines)


def load_golden_cases(data_dir: str | Path | None = None) -> list[dict]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    path = Path(data_dir) / "golden_cases.json"
    return json.loads(path.read_text())


def run_evals(investigate: Callable[[dict], object], cases: list[dict] | None = None) -> EvalResult:
    """``investigate`` takes an incident dict and returns an object with
    ``root_cause`` and ``evidence`` attributes (or a dict with those
    keys)."""
    cases = cases if cases is not None else load_golden_cases()
    results = EvalResult()
    for case in cases:
        incident = case["incident"]
        report = investigate(incident)
        if isinstance(report, dict):
            predicted = report.get("root_cause", "")
            evidence = report.get("evidence", [])
        else:
            predicted = getattr(report, "root_cause", "")
            evidence = getattr(report, "evidence", [])
        expected = case["expected_root_cause"]
        keywords = case.get("expected_evidence_keywords", [])
        blob = " ".join(evidence).lower()
        hits = sum(1 for kw in keywords if kw.lower() in blob)
        evidence_score = hits / len(keywords) if keywords else 0.0
        results.cases.append(
            CaseResult(
                case_id=case["id"],
                expected=expected,
                predicted=predicted,
                correct=predicted == expected,
                evidence_score=evidence_score,
                false_alarm=(expected == "no-fault" and predicted != "no-fault"),
            )
        )
    return results


def main() -> None:
    """Standalone eval run: builds the demo agent and scores it."""
    from network_intelligence.agent import build_agent
    from network_intelligence.config import ServingConfig, StoreConfig
    from network_intelligence.retrieval import default_retriever
    from network_intelligence.serving import get_backend
    from network_intelligence.stores import build_stores
    from network_intelligence.telemetry import TelemetryGenerator

    metrics, incidents, _ = build_stores(StoreConfig.from_env())
    metrics.load(TelemetryGenerator(seed=42).generate())
    incidents.load(TelemetryGenerator(seed=42).generate())
    agent = build_agent(
        get_backend(ServingConfig.from_env()),
        metrics,
        incidents,
        retriever=default_retriever(),
    )
    result = run_evals(agent.investigate)
    print(result.render_text())


if __name__ == "__main__":
    main()
