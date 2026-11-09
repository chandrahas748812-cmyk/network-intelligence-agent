"""Metric / log / alert / incident stores.

The demo runs on in-memory stores — no infrastructure needed. When
``DATABASE_URL`` / ``REDIS_URL`` are set and the optional drivers are
installed, the Postgres / Redis implementations are used instead; if the
connection fails (or the driver is missing) the store logs a warning and
falls back to memory. The fallback is deliberate: the demo must run on a
laptop with no Docker running.

Postgres schema (created by ``docker-compose.yml`` / ``init.sql``):
    metrics(ts TIMESTAMPTZ, device TEXT, metric TEXT, value DOUBLE PRECISION)
    logs(ts TIMESTAMPTZ, device TEXT, level TEXT, message TEXT)
    alerts(id TEXT PRIMARY KEY, ts TIMESTAMPTZ, device TEXT, rule TEXT,
           severity TEXT, message TEXT)
    incidents(id TEXT PRIMARY KEY, title TEXT, device TEXT, severity TEXT,
              started_at TIMESTAMPTZ, signals JSONB, expected_root_cause TEXT,
              status TEXT)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from network_intelligence.config import StoreConfig
from network_intelligence.telemetry import Alert, Incident, LogLine, TelemetryPoint

log = logging.getLogger(__name__)


class InMemoryMetricsStore:
    """Metrics + logs + alerts in process memory."""

    def __init__(self) -> None:
        self.points: list[dict] = []
        self.logs: list[dict] = []
        self.alerts: list[dict] = []

    def load(self, bundle: dict) -> None:
        self.points = list(bundle.get("metrics", []))
        self.logs = list(bundle.get("logs", []))
        self.alerts = list(bundle.get("alerts", []))

    def query_metrics(self, device: str, metric: str, minutes: int = 30) -> list[dict]:
        rows = [p for p in self.points if p["device"] == device and p["metric"] == metric]
        return rows[-minutes:] if minutes else rows

    def latest_value(self, device: str, metric: str) -> float | None:
        rows = self.query_metrics(device, metric, minutes=0)
        return rows[-1]["value"] if rows else None

    def search_logs(self, device: str, level: str | None = None, limit: int = 50) -> list[dict]:
        rows = [l for l in self.logs if l["device"] == device]
        if level:
            rows = [l for l in rows if l["level"] == level]
        return rows[-limit:]

    def active_alerts(self, device: str | None = None) -> list[dict]:
        if device:
            return [a for a in self.alerts if a["device"] == device]
        return list(self.alerts)


class InMemoryIncidentStore:
    def __init__(self) -> None:
        self.incidents: dict[str, dict] = {}

    def load(self, bundle: dict) -> None:
        for inc in bundle.get("incidents", []):
            self.incidents[inc["id"]] = dict(inc)

    def get(self, incident_id: str) -> dict | None:
        return self.incidents.get(incident_id)

    def list_open(self) -> list[dict]:
        return [i for i in self.incidents.values() if i.get("status") == "open"]

    def save(self, incident: dict) -> None:
        self.incidents[incident["id"]] = dict(incident)


class DedupCache:
    """Idempotency / dedup cache for tool calls and watcher alerts.

    Redis when ``REDIS_URL`` is set and reachable, else in-memory with TTL.
    """

    def __init__(self, redis_url: str = "") -> None:
        self._mem: dict[str, tuple[float, Any]] = {}
        self._redis = None
        if redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=2)
                self._redis.ping()
                log.info("DedupCache: using Redis at %s", redis_url)
            except Exception as exc:  # noqa: BLE001 - fall back, never crash demo
                log.warning("DedupCache: Redis unavailable (%s); using in-memory cache", exc)
                self._redis = None

    def get(self, key: str) -> Any | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception:  # noqa: BLE001
                return self._mem_get(key)
        return self._mem_get(key)

    def _mem_get(self, key: str) -> Any | None:
        item = self._mem.get(key)
        if item is None:
            return None
        expires, value = item
        if time.time() > expires:
            del self._mem[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_s: int = 300) -> None:
        if self._redis is not None:
            try:
                self._redis.set(key, json.dumps(value), ex=ttl_s)
                return
            except Exception:  # noqa: BLE001
                pass
        self._mem[key] = (time.time() + ttl_s, value)

    def seen(self, key: str, ttl_s: int = 300) -> bool:
        """True if ``key`` was already recorded (and not expired)."""
        if self.get(key) is not None:
            return True
        self.set(key, {"seen_at": time.time()}, ttl_s=ttl_s)
        return False


class PostgresMetricsStore(InMemoryMetricsStore):
    """Postgres-backed metrics store. Falls back to memory on any error.

    Only used when ``DATABASE_URL`` is set. The driver (``psycopg``) is an
    optional dependency — import failure also falls back cleanly.
    """

    def __init__(self, database_url: str) -> None:
        super().__init__()
        self._conn = None
        self.database_url = database_url
        try:
            import psycopg  # type: ignore

            self._conn = psycopg.connect(database_url, connect_timeout=3)
            log.info("PostgresMetricsStore: connected")
        except Exception as exc:  # noqa: BLE001
            log.warning("PostgresMetricsStore: unavailable (%s); using in-memory store", exc)
            self._conn = None

    @property
    def live(self) -> bool:
        return self._conn is not None

    def query_metrics(self, device: str, metric: str, minutes: int = 30) -> list[dict]:
        if self._conn is None:
            return super().query_metrics(device, metric, minutes)
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT ts, device, metric, value FROM metrics "
                    "WHERE device=%s AND metric=%s ORDER BY ts DESC LIMIT %s",
                    (device, metric, minutes),
                )
                return [
                    {"ts": str(r[0]), "device": r[1], "metric": r[2], "value": r[3]}
                    for r in reversed(cur.fetchall())
                ]
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres query failed (%s); falling back to memory", exc)
            return super().query_metrics(device, metric, minutes)


def build_stores(cfg: StoreConfig) -> tuple[InMemoryMetricsStore, InMemoryIncidentStore, DedupCache]:
    """Build the stores the demo/API use, with graceful fallbacks."""
    if cfg.database_url:
        metrics = PostgresMetricsStore(cfg.database_url)
    else:
        metrics = InMemoryMetricsStore()
    incidents = InMemoryIncidentStore()
    cache = DedupCache(cfg.redis_url)
    return metrics, incidents, cache


# Re-export telemetry dataclasses for convenience in tests.
__all__ = [
    "Alert",
    "DedupCache",
    "Incident",
    "InMemoryIncidentStore",
    "InMemoryMetricsStore",
    "LogLine",
    "PostgresMetricsStore",
    "TelemetryPoint",
    "build_stores",
]
