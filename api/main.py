"""FastAPI serving layer.

Endpoints
---------
* ``GET  /health`` — liveness + backend label (mock is labelled mock).
* ``POST /generate`` — LLM completion with sampling + serving config.
  Returns tokens/s and ``simulated: true`` when the mock backend ran.
* ``POST /investigate`` — run the investigation agent on an incident
  and return the structured diagnostic report.
* ``POST /watcher/scan`` — one background-watcher pass over telemetry.
* ``GET  /eval`` — run the golden-case eval harness, return scores.
* ``GET  /models`` — model registry contents.
* ``POST /throughput/diagnose`` — the "why 3 tok/s?" check.

Run: ``uvicorn api.main:app --port 8000`` (from the repo root).
The demo (``demo.py``) does not need the server — it calls the agent
directly — but the API is tested with FastAPI's TestClient.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from network_intelligence.agent import build_agent
from network_intelligence.config import (
    AgentConfig,
    ServingConfig,
    StoreConfig,
    WatcherConfig,
)
from network_intelligence.evals import load_golden_cases, run_evals
from network_intelligence.registry import ModelRegistry
from network_intelligence.retrieval import default_retriever
from network_intelligence.serving import (
    diagnose_throughput,
    get_backend,
    model_tokens_per_s,
)
from network_intelligence.stores import build_stores
from network_intelligence.telemetry import TelemetryGenerator
from network_intelligence.watcher import TelemetryWatcher

log = logging.getLogger(__name__)
app = FastAPI(title="Network Intelligence Agent", version="0.1.0")

# --- shared singletons (seeded demo data; swap stores via env) --------
_serving_cfg = ServingConfig.from_env()
_metrics, _incidents, _cache = build_stores(StoreConfig.from_env())
_seed = TelemetryGenerator(seed=42).generate()
_metrics.load(_seed)
_incidents.load(_seed)
_backend = get_backend(_serving_cfg)
_agent = build_agent(_backend, _metrics, _incidents, retriever=default_retriever())
_watcher = TelemetryWatcher(_metrics, WatcherConfig.from_env(), _cache)
_registry = ModelRegistry()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int | None = None
    serving: dict = Field(default_factory=dict)


class InvestigateRequest(BaseModel):
    id: str
    title: str
    device: str
    severity: str = "warning"
    signals: dict = Field(default_factory=dict)
    expected_root_cause: str = ""


class ThroughputRequest(BaseModel):
    measured_tok_s: float
    kv_cache_usage_pct: float = 55.0
    batch_size: int | None = None
    speculative_accept_rate: float | None = None
    serving: dict = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": _serving_cfg.backend,
        "model": _serving_cfg.model,
        "simulated": _serving_cfg.backend == "mock",
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    cfg = ServingConfig(**{**_serving_cfg.__dict__, **req.serving}) if req.serving else _serving_cfg
    backend = get_backend(cfg) if req.serving else _backend
    return backend.generate(req.prompt, max_tokens=req.max_tokens).to_dict()


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> dict:
    incident = req.model_dump()
    return _agent.investigate(incident).to_dict()


@app.post("/watcher/scan")
def watcher_scan() -> dict:
    alerts = _watcher.scan()
    return {"alerts": [a.to_dict() for a in alerts], "count": len(alerts)}


@app.get("/eval")
def eval_endpoint() -> dict:
    return run_evals(_agent.investigate, load_golden_cases()).to_dict()


@app.get("/models")
def models() -> dict:
    return {"versions": [v.to_dict() for v in _registry.versions.values()]}


@app.post("/throughput/diagnose")
def throughput_diagnose(req: ThroughputRequest) -> dict:
    cfg = ServingConfig(**{**_serving_cfg.__dict__, **req.serving}) if req.serving else _serving_cfg
    diag = diagnose_throughput(
        cfg,
        measured_tok_s=req.measured_tok_s,
        kv_cache_usage_pct=req.kv_cache_usage_pct,
        batch_size=req.batch_size,
        speculative_accept_rate=req.speculative_accept_rate,
    )
    return diag.to_dict()


@app.get("/throughput/expected")
def throughput_expected() -> dict:
    return {
        "model": _serving_cfg.model,
        "expected_tok_s": round(model_tokens_per_s(_serving_cfg), 2),
        "note": "Heuristic model for triage, not a measured GPU benchmark.",
    }
