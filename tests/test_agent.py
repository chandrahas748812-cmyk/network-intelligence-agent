"""Agent loop tests: the right root cause, with evidence, for each
seeded incident — and an honest 'no-fault' when there is none."""

from network_intelligence.tools import ToolRunner


def test_investigate_kv_cache_exhaustion(agent, incident_store):
    incident = incident_store.get("INC-1001")
    report = agent.investigate(incident)
    assert report.root_cause == "kv_cache_exhaustion"
    assert report.confidence >= 0.6
    assert report.playbook_id == "pb-throughput-collapse"
    assert any("kv_cache" in e for e in report.evidence)
    assert len(report.trace) == 5  # the full plan: metrics, logs, alerts, correlate, tree
    assert report.trace[0]["tool"] == "query_metrics"
    assert report.trace[-1]["tool"] == "decision_tree"


def test_investigate_faulty_optic(agent, incident_store):
    report = agent.investigate(incident_store.get("INC-1002"))
    assert report.root_cause == "faulty_optic"
    assert report.playbook_id == "pb-packet-loss"
    assert any("CRC" in e for e in report.evidence)


def test_investigate_batching_too_small(agent, incident_store):
    report = agent.investigate(incident_store.get("INC-1003"))
    assert report.root_cause == "batching_too_small"
    assert report.playbook_id == "pb-gpu-underutilised"


def test_investigate_healthy_is_no_fault(agent):
    incident = {
        "id": "INC-1004",
        "title": "Elevated p99 latency, no errors",
        "device": "gpu-node-01",
        "severity": "info",
        "signals": {"p99_latency_ms": 260.0, "gpu_util_pct": 74.0, "inference_tok_per_s": 17.5},
    }
    report = agent.investigate(incident)
    assert report.root_cause == "no-fault"
    assert "no fault" in report.summary.lower() or "no-fault" in report.summary


def test_report_render_text(agent, incident_store):
    report = agent.investigate(incident_store.get("INC-1001"))
    text = report.render_text()
    assert "kv_cache_exhaustion" in text
    assert "Recommended action" in text


def test_tool_runner_retries_flaky_tool():
    calls = {"n": 0}

    def flaky(device: str):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient failure")
        return {"ok": True, "evidence": ["flaky recovered"]}

    runner = ToolRunner()
    result = runner.call("flaky_tool", flaky, {"device": "gpu-node-01"})
    assert result.error is None
    assert result.attempts == 2
    assert result.output == {"ok": True, "evidence": ["flaky recovered"]}


def test_tool_runner_dedups_identical_calls():
    calls = {"n": 0}

    def counting(device: str):
        calls["n"] += 1
        return {"device": device, "evidence": ["once"]}

    runner = ToolRunner()
    first = runner.call("counting", counting, {"device": "gpu-node-01"})
    second = runner.call("counting", counting, {"device": "gpu-node-01"})
    assert calls["n"] == 1  # second call served from cache
    assert second.cached is True
    assert first.output == second.output


def test_tool_runner_reports_failure_after_retries():
    def always_fails(device: str):
        raise RuntimeError("boom")

    runner = ToolRunner()
    result = runner.call("broken", always_fails, {"device": "gpu-node-01"})
    assert result.error == "boom"
    assert result.attempts == runner.cfg.max_retries
