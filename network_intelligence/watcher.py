"""Background watcher agents: flag problems before anyone asks.

The watcher polls the metrics store on a schedule (or once, on demand)
and flags anomalies two ways:

* **hard thresholds** — e.g. packet loss > 1%, KV cache > 90%;
* **rolling z-scores** — the latest value vs the recent mean/stddev, so
  slow drift and sudden jumps are both caught.

Findings are deduplicated through the :class:`DedupCache` (Redis when
configured, in-memory otherwise) so the same anomaly does not page
twice. A watcher finding is a *candidate* incident: it carries the
evidence and a suggested playbook, and the investigation agent takes
it from there. The watcher never pages on a single noisy point — the
z-score needs a minimum window first.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from network_intelligence.config import WatcherConfig
from network_intelligence.stores import DedupCache, InMemoryMetricsStore


@dataclass
class WatcherAlert:
    id: str
    ts: str
    device: str
    metric: str
    latest: float
    z_score: float | None
    threshold_breached: bool
    message: str
    suggested_playbook: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "device": self.device,
            "metric": self.metric,
            "latest": self.latest,
            "z_score": round(self.z_score, 2) if self.z_score is not None else None,
            "threshold_breached": self.threshold_breached,
            "message": self.message,
            "suggested_playbook": self.suggested_playbook,
        }


_PLAYBOOK_BY_METRIC = {
    "inference_tok_per_s": "pb-throughput-collapse",
    "kv_cache_usage_pct": "pb-throughput-collapse",
    "p99_latency_ms": "pb-throughput-collapse",
    "packet_loss_pct": "pb-packet-loss",
    "interface_errors": "pb-packet-loss",
    "gpu_util_pct": "pb-gpu-underutilised",
}


class TelemetryWatcher:
    def __init__(
        self,
        metrics: InMemoryMetricsStore,
        cfg: WatcherConfig | None = None,
        cache: DedupCache | None = None,
    ) -> None:
        self.metrics = metrics
        self.cfg = cfg or WatcherConfig()
        self.cache = cache or DedupCache()

    def scan(self, devices: list[str] | None = None) -> list[WatcherAlert]:
        """One pass over the store. Returns new (not yet deduped) alerts."""
        found: list[WatcherAlert] = []
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        device_names = devices or sorted({p["device"] for p in self.metrics.points})
        for device in device_names:
            metric_names = sorted({p["metric"] for p in self.metrics.points if p["device"] == device})
            for metric in metric_names:
                rows = self.metrics.query_metrics(device, metric, minutes=0)
                if len(rows) < 2:
                    continue
                values = [r["value"] for r in rows]
                latest = values[-1]
                z: float | None = None
                if len(values) >= self.cfg.min_window:
                    # Reference window: the points BEFORE the most recent
                    # window. Comparing against the recent window itself
                    # would hide a sustained anomaly (the window is
                    # already anomalous); the reference window is the
                    # healthy baseline the latest value is judged against.
                    # Use a wide reference (4x the window) so the
                    # standard-deviation estimate is stable and healthy
                    # jitter does not page.
                    if len(values) >= 5 * self.cfg.min_window:
                        window = values[-5 * self.cfg.min_window:-self.cfg.min_window]
                    else:
                        window = values[:-1]
                    mean = statistics.fmean(window)
                    sd = statistics.pstdev(window) if len(window) > 1 else 0.0
                    z = (latest - mean) / sd if sd > 0 else 0.0
                hard = self.cfg.thresholds.get(metric)
                breached = hard is not None and latest > hard
                if z is not None and abs(z) < self.cfg.z_threshold and not breached:
                    continue
                if z is None and not breached:
                    continue
                key = f"watcher:{device}:{metric}:{round(latest, 1)}"
                if self.cache.seen(key, ttl_s=3600):
                    continue  # already flagged within the hour
                if breached:
                    msg = f"{metric} on {device} = {latest} (threshold {hard})"
                else:
                    msg = f"{metric} on {device} = {latest} (z={z:.1f} vs recent mean)"
                found.append(
                    WatcherAlert(
                        id=f"WA-{device}-{metric}",
                        ts=now,
                        device=device,
                        metric=metric,
                        latest=latest,
                        z_score=z,
                        threshold_breached=breached,
                        message=msg,
                        suggested_playbook=_PLAYBOOK_BY_METRIC.get(metric, "pb-generic"),
                    )
                )
        return found
