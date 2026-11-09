"""Playbook + tool tests: selection, decision tree, thresholds,
event correlation."""

from network_intelligence.playbooks import playbook_for_root_cause, select_playbook
from network_intelligence.tools import make_tools


def test_select_playbook_throughput():
    pb = select_playbook({"signals": {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0}})
    assert pb.id == "pb-throughput-collapse"


def test_select_playbook_packet_loss():
    pb = select_playbook({"signals": {"packet_loss_pct": 2.3, "interface_errors": 41.0}})
    assert pb.id == "pb-packet-loss"


def test_select_playbook_gpu_underutilised():
    pb = select_playbook({"signals": {"gpu_util_pct": 12.0, "inference_tok_per_s": 6.5}})
    assert pb.id == "pb-gpu-underutilised"


def test_select_playbook_falls_back_to_generic():
    pb = select_playbook({"signals": {"mystery_metric": 1.0}})
    assert pb.id == "pb-generic"


def test_playbook_for_root_cause():
    assert playbook_for_root_cause("kv_cache_exhaustion").id == "pb-throughput-collapse"
    assert playbook_for_root_cause("faulty_optic").id == "pb-packet-loss"
    assert playbook_for_root_cause("batching_too_small").id == "pb-gpu-underutilised"
    assert playbook_for_root_cause("unknown") is None


def test_decision_tree_branches(metrics_store, incident_store):
    tools = make_tools(metrics_store, incident_store)
    tree = tools["decision_tree"]
    assert tree("gpu-node-02", {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0})["branch"] == "kv_cache_exhaustion"
    assert tree("leaf-01", {"packet_loss_pct": 2.3})["branch"] == "faulty_optic"
    assert tree("gpu-node-03", {"gpu_util_pct": 12.0, "inference_tok_per_s": 6.5})["branch"] == "batching_too_small"
    assert tree("gpu-node-01", {"p99_latency_ms": 260.0})["branch"] == "no-fault"


def test_threshold_check(metrics_store, incident_store):
    tools = make_tools(metrics_store, incident_store)
    breached = tools["threshold_check"]("gpu-node-02", "kv_cache_usage_pct", 90.0, "gt")
    assert breached["breached"] is True
    assert breached["latest"] >= 90.0
    ok = tools["threshold_check"]("gpu-node-01", "kv_cache_usage_pct", 90.0, "gt")
    assert ok["breached"] is False


def test_correlate_events_finds_overlap(metrics_store, incident_store):
    tools = make_tools(metrics_store, incident_store)
    out = tools["correlate_events"]("gpu-node-02", minutes=30)
    assert out["correlated"] is True
    assert len(out["events"]) >= 2  # at least one alert + one error log


def test_query_metrics_stats(metrics_store, incident_store):
    tools = make_tools(metrics_store, incident_store)
    out = tools["query_metrics"]("gpu-node-02", ["inference_tok_per_s", "kv_cache_usage_pct"], minutes=30)
    assert out["metrics"]["inference_tok_per_s"]["latest"] < 8.0
    assert out["metrics"]["kv_cache_usage_pct"]["latest"] > 90.0
    assert out["metrics"]["inference_tok_per_s"]["points"] == 30
