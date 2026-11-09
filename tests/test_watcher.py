"""Watcher tests: anomalies get flagged, healthy telemetry does not
page on hard thresholds, and repeats are deduped."""

from network_intelligence.stores import InMemoryMetricsStore
from network_intelligence.telemetry import TelemetryGenerator, healthy_snapshot
from network_intelligence.watcher import TelemetryWatcher


def test_watcher_flags_seeded_anomalies(metrics_store):
    alerts = TelemetryWatcher(metrics_store).scan()
    flagged = {(a.device, a.metric) for a in alerts}
    assert ("gpu-node-02", "kv_cache_usage_pct") in flagged
    assert ("gpu-node-02", "inference_tok_per_s") in flagged
    assert ("leaf-01", "packet_loss_pct") in flagged
    assert ("gpu-node-03", "gpu_util_pct") in flagged
    # Every alert names the playbook that should investigate it.
    by_key = {(a.device, a.metric): a for a in alerts}
    assert by_key[("gpu-node-02", "kv_cache_usage_pct")].suggested_playbook == "pb-throughput-collapse"
    assert by_key[("leaf-01", "packet_loss_pct")].suggested_playbook == "pb-packet-loss"


def test_watcher_no_hard_threshold_breach_on_healthy_data():
    store = InMemoryMetricsStore()
    store.load(healthy_snapshot())
    alerts = TelemetryWatcher(store).scan()
    assert all(not a.threshold_breached for a in alerts)


def test_watcher_dedups_repeats(metrics_store):
    watcher = TelemetryWatcher(metrics_store)
    first = watcher.scan()
    assert len(first) > 0
    second = watcher.scan()
    assert second == []  # same anomalies already flagged within the hour


def test_watcher_flags_a_constructed_spike():
    store = InMemoryMetricsStore()
    points = []
    for i in range(30):
        points.append({"ts": f"2026-11-08T00:{i:02d}:00Z", "device": "gpu-node-01",
                       "metric": "p99_latency_ms", "value": 180.0})
    points.append({"ts": "2026-11-08T00:30:00Z", "device": "gpu-node-01",
                   "metric": "p99_latency_ms", "value": 4200.0})
    store.points = points
    alerts = TelemetryWatcher(store).scan()
    assert len(alerts) == 1
    assert alerts[0].device == "gpu-node-01"
    assert alerts[0].metric == "p99_latency_ms"
    assert alerts[0].threshold_breached is True
