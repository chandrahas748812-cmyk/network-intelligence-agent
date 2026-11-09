"""One-command demo: seed -> watch -> investigate -> report -> eval.

Usage (from the repo root, no Docker, no GPU needed):

    python demo.py

What it does:
1. Seeds synthetic telemetry (60 minutes, 3 incident scenarios).
2. Runs one background-watcher pass and prints flagged anomalies.
3. Runs the investigation agent end-to-end on the throughput-collapse
   incident (the JD's "3 tok/s" case) and prints the diagnostic report.
4. Runs the throughput diagnostic ("why 3 tok/s?").
5. Curates one fine-tuning example from the investigation trace.
6. Runs the golden-case eval harness and prints the scores.

Everything runs on CPU with the mock backend. Numbers labelled
"simulated" come from the throughput heuristic, not a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from network_intelligence.agent import build_agent  # noqa: E402
from network_intelligence.config import AgentConfig, ServingConfig, StoreConfig  # noqa: E402
from network_intelligence.evals import load_golden_cases, run_evals  # noqa: E402
from network_intelligence.finetune import trace_to_example  # noqa: E402
from network_intelligence.retrieval import default_retriever  # noqa: E402
from network_intelligence.serving import diagnose_throughput  # noqa: E402
from network_intelligence.stores import build_stores  # noqa: E402
from network_intelligence.telemetry import TelemetryGenerator  # noqa: E402
from network_intelligence.watcher import TelemetryWatcher  # noqa: E402


def main() -> int:
    print("== Network Intelligence Agent — demo (mock backend, CPU) ==\n")

    # 1. Seed telemetry.
    bundle = TelemetryGenerator(seed=42).generate(minutes=60)
    metrics, incidents, cache = build_stores(StoreConfig())
    metrics.load(bundle)
    incidents.load(bundle)
    print(f"[1] Seeded {len(bundle['metrics'])} metric points, "
          f"{len(bundle['logs'])} log lines, {len(bundle['alerts'])} alerts, "
          f"{len(bundle['incidents'])} incidents.\n")

    # 2. Watcher pass.
    watcher = TelemetryWatcher(metrics, cache=cache)
    alerts = watcher.scan()
    print(f"[2] Watcher pass: {len(alerts)} new anomaly flag(s).")
    for a in alerts:
        print(f"    - {a.message}  -> playbook {a.suggested_playbook}")
    print()

    # 3. Investigate the throughput-collapse incident end to end.
    agent = build_agent(
        __import__("network_intelligence.serving", fromlist=["get_backend"]).get_backend(ServingConfig()),
        metrics, incidents, retriever=default_retriever(), cfg=AgentConfig(),
    )
    incident = incidents.get("INC-1001")
    print(f"[3] Investigating {incident['id']}: {incident['title']} on {incident['device']}")
    report = agent.investigate(incident)
    print(report.render_text())
    print("\n    Tool trace:")
    for step in report.trace:
        print(f"    - {step['tool']} (attempts={step['attempts']}, cached={step['cached']})"
              f"{' ERROR: ' + step['error'] if step['error'] else ''}")
    print()

    # 4. Throughput diagnostic — the JD's "why 3 tok/s?" question.
    cfg = ServingConfig(max_num_seqs=64, kv_cache_gb=16.0)
    diag = diagnose_throughput(cfg, measured_tok_s=3.1, kv_cache_usage_pct=97.0, batch_size=1)
    print("[4] Throughput diagnostic (measured 3.1 tok/s, KV cache 97%):")
    print(f"    expected (heuristic): {diag.expected_tok_s:.1f} tok/s, ratio {diag.ratio:.2f}")
    for cause in diag.likely_causes:
        print(f"    - {cause['cause']} (confidence {cause['confidence']:.1f}): {cause['detail']}")
    print(f"    -> {diag.recommendation}\n")

    # 5. Fine-tuning curation from this investigation's trace.
    trace = {
        "incident": incident,
        "steps": [
            {"tool": s["tool"], "args": s["args"], "observation": s["output_summary"]}
            for s in report.trace
        ],
        "diagnosis": {
            "root_cause": report.root_cause,
            "evidence": report.evidence,
            "recommended_action": report.recommended_action,
        },
    }
    example = trace_to_example(trace)
    print("[5] Fine-tuning example curated from the trace (first 400 chars):")
    print("    " + json.dumps(example.to_dict())[:400] + "…\n")

    # 6. Eval harness.
    result = run_evals(agent.investigate, load_golden_cases())
    print("[6] Eval harness (golden cases):")
    print(result.render_text())
    print("\nDemo done. Run the API with:  uvicorn api.main:app --port 8000")
    print("Run the tests with:           pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
