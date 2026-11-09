-- Schema for the Postgres metrics store (see stores.py).
CREATE TABLE IF NOT EXISTS metrics (
    ts TIMESTAMPTZ NOT NULL,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_device_metric_ts
    ON metrics (device, metric, ts);

CREATE TABLE IF NOT EXISTS logs (
    ts TIMESTAMPTZ NOT NULL,
    device TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_device_ts ON logs (device, ts);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    device TEXT NOT NULL,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    device TEXT NOT NULL,
    severity TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    signals JSONB NOT NULL DEFAULT '{}',
    expected_root_cause TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open'
);
