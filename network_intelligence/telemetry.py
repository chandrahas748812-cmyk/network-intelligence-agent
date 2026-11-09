"""Synthetic telemetry generator.

Produces deterministic, seeded telemetry for a small fictional inference
fleet: GPU inference nodes, spine/leaf network devices, and the metrics,
logs, alerts, and incidents an investigation agent would work with.

Everything here is invented. Device names, values, and incident scenarios
are placeholders for a demo — they do not describe any real Tesla system.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class TelemetryPoint:
    ts: str  # ISO timestamp
    device: str
    metric: str
    value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogLine:
    ts: str
    device: str
    level: str  # INFO | WARN | ERROR
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    id: str
    ts: str
    device: str
    rule: str
    severity: str  # info | warning | critical
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Incident:
    id: str
    title: str
    device: str
    severity: str
    started_at: str
    signals: dict
    expected_root_cause: str = ""
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)


# Fictional fleet. GPU nodes serve LLM inference; spine/leaf carry traffic.
DEVICES = {
    "gpu-node-01": {"kind": "gpu", "metrics": ["inference_tok_per_s", "p99_latency_ms", "gpu_util_pct", "kv_cache_usage_pct", "gpu_temp_c"]},
    "gpu-node-02": {"kind": "gpu", "metrics": ["inference_tok_per_s", "p99_latency_ms", "gpu_util_pct", "kv_cache_usage_pct", "gpu_temp_c"]},
    "gpu-node-03": {"kind": "gpu", "metrics": ["inference_tok_per_s", "p99_latency_ms", "gpu_util_pct", "kv_cache_usage_pct", "gpu_temp_c"]},
    "spine-01": {"kind": "network", "metrics": ["packet_loss_pct", "interface_errors", "p99_latency_ms"]},
    "leaf-01": {"kind": "network", "metrics": ["packet_loss_pct", "interface_errors", "p99_latency_ms"]},
    "leaf-02": {"kind": "network", "metrics": ["packet_loss_pct", "interface_errors", "p99_latency_ms"]},
}

# Healthy baselines (mean, stddev) per metric.
BASELINES = {
    "inference_tok_per_s": (18.0, 1.5),
    "p99_latency_ms": (180.0, 25.0),
    "gpu_util_pct": (72.0, 8.0),
    "kv_cache_usage_pct": (55.0, 8.0),
    "gpu_temp_c": (68.0, 3.0),
    "packet_loss_pct": (0.02, 0.02),
    "interface_errors": (0.5, 0.8),
}

# Seeded incident scenarios. Each injects a recognisable signature into
# the metrics/logs so the agent's diagnosis can be scored against the
# expected root cause.
SCENARIOS = {
    "kv-cache-exhaustion": {
        "device": "gpu-node-02",
        "title": "LLM inference throughput collapsed to ~3 tok/s",
        "severity": "critical",
        "signals": {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0, "p99_latency_ms": 4200.0, "gpu_util_pct": 34.0},
        "overrides": {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0, "p99_latency_ms": 4200.0, "gpu_util_pct": 34.0},
        "log_templates": [
            "ERROR kv_cache: allocation failed, KV cache at 97% — falling back to batch=1",
            "WARN scheduler: max_num_seqs=64 requested, only 1 sequence fits in KV cache",
            "ERROR serving: decode step 3.1 tok/s, expected ~18 tok/s for this model",
        ],
        "alerts": [("KVCacheExhaustion", "critical", "KV cache usage 97% on gpu-node-02")],
        "expected_root_cause": "kv_cache_exhaustion",
    },
    "faulty-optic": {
        "device": "leaf-01",
        "title": "Packet loss between leaf-01 and spine-01",
        "severity": "critical",
        "signals": {"packet_loss_pct": 2.3, "interface_errors": 41.0, "p99_latency_ms": 640.0},
        "overrides": {"packet_loss_pct": 2.3, "interface_errors": 41.0, "p99_latency_ms": 640.0},
        "log_templates": [
            "ERROR interface eth1: CRC errors rising, 41 in last 5m (baseline <1)",
            "WARN bgp: flapping on leaf-01 uplink to spine-01, 6 flaps in 10m",
            "ERROR link-monitor: packet loss 2.3% on leaf-01 -> spine-01",
        ],
        "alerts": [("PacketLossHigh", "critical", "Packet loss 2.3% on leaf-01")],
        "expected_root_cause": "faulty_optic",
    },
    "underutilised-gpu": {
        "device": "gpu-node-03",
        "title": "GPU underutilised while request queue grows",
        "severity": "warning",
        "signals": {"gpu_util_pct": 12.0, "inference_tok_per_s": 6.5, "p99_latency_ms": 950.0},
        "overrides": {"gpu_util_pct": 12.0, "inference_tok_per_s": 6.5, "p99_latency_ms": 950.0},
        "log_templates": [
            "WARN scheduler: max_num_seqs=8, GPU at 12% util with 140 requests queued",
            "INFO autoscaler: no scale-up triggered, single GPU serving",
            "WARN serving: batch size 2-4, decode is memory-bandwidth bound but batches are tiny",
        ],
        "alerts": [("GPUUnderutilised", "warning", "GPU util 12% with queue depth 140 on gpu-node-03")],
        "expected_root_cause": "batching_too_small",
    },
}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TelemetryGenerator:
    """Deterministic synthetic telemetry. Same seed => same data."""

    def __init__(self, seed: int = 42, start: datetime | None = None):
        self.rng = random.Random(seed)
        self.start = start or (datetime.now(timezone.utc) - timedelta(minutes=60))

    def generate(
        self,
        minutes: int = 60,
        step_seconds: int = 60,
        scenarios: list[str] | None = None,
    ) -> dict:
        """Generate metrics, logs, alerts and incidents.

        ``scenarios`` names the incident scenarios to inject (default: all
        three). The last 10 minutes of each scenario's device carry the
        anomalous values; earlier points follow the healthy baseline.
        """
        scenarios = scenarios if scenarios is not None else list(SCENARIOS)
        points: list[TelemetryPoint] = []
        logs: list[LogLine] = []
        alerts: list[Alert] = []
        incidents: list[Incident] = []

        steps = max(1, minutes * 60 // step_seconds)
        anomaly_start = steps - 10  # anomalies in the final 10 steps

        for i in range(steps):
            ts = _iso(self.start + timedelta(seconds=i * step_seconds))
            for device, spec in DEVICES.items():
                for metric in spec["metrics"]:
                    mean, sd = BASELINES[metric]
                    value = max(0.0, self.rng.gauss(mean, sd))
                    # Inject the scenario signature on its device.
                    for name in scenarios:
                        sc = SCENARIOS[name]
                        if sc["device"] == device and metric in sc["overrides"] and i >= anomaly_start:
                            value = sc["overrides"][metric] * self.rng.uniform(0.95, 1.05)
                    points.append(TelemetryPoint(ts, device, metric, round(value, 2)))

            # A few routine log lines per step.
            if i % 5 == 0:
                device = self.rng.choice(list(DEVICES))
                logs.append(LogLine(ts, device, "INFO", f"health check ok on {device}"))

        # Incident-specific logs / alerts / incident records.
        for n, name in enumerate(scenarios):
            sc = SCENARIOS[name]
            inc_ts = _iso(self.start + timedelta(seconds=anomaly_start * step_seconds))
            for j, tmpl in enumerate(sc["log_templates"]):
                level = tmpl.split(" ", 1)[0]
                logs.append(
                    LogLine(_iso(self.start + timedelta(seconds=(anomaly_start + j) * step_seconds)), sc["device"], level, tmpl.split(" ", 1)[1])
                )
            for k, (rule, sev, msg) in enumerate(sc["alerts"]):
                alerts.append(Alert(f"AL-{1000 + n * 10 + k}", inc_ts, sc["device"], rule, sev, msg))
            incidents.append(
                Incident(
                    id=f"INC-{1001 + n}",
                    title=sc["title"],
                    device=sc["device"],
                    severity=sc["severity"],
                    started_at=inc_ts,
                    signals=dict(sc["signals"]),
                    expected_root_cause=sc["expected_root_cause"],
                )
            )

        logs.sort(key=lambda l: l.ts)
        return {
            "metrics": [p.to_dict() for p in points],
            "logs": [l.to_dict() for l in logs],
            "alerts": [a.to_dict() for a in alerts],
            "incidents": [i.to_dict() for i in incidents],
        }

    def to_json(self, minutes: int = 60, **kwargs) -> str:
        return json.dumps(self.generate(minutes=minutes, **kwargs), indent=2)


def healthy_snapshot(seed: int = 7, minutes: int = 30) -> dict:
    """Telemetry with no injected scenario — used for false-alarm evals."""
    gen = TelemetryGenerator(seed=seed)
    return gen.generate(minutes=minutes, scenarios=[])
