"""LLM serving layer: pluggable backends + throughput diagnostics.

Backends
--------
* ``MockBackend`` — deterministic, CPU-only stand-in. It answers
  ``generate`` with templated text and *simulated* latency / token
  counts derived from the serving config. It is labelled as a mock
  everywhere it appears; it is NOT a vLLM benchmark.
* ``VLLMBackend`` — thin wrapper around a real vLLM engine. Importing
  ``vllm`` is optional; if it is not installed the backend raises a
  clear error instead of pretending to serve. GPU runs only.
* ``OpenAICompatibleBackend`` — posts to any OpenAI-compatible
  ``/chat/completions`` endpoint (vLLM server, TGI, etc.) when
  ``NIA_LLM_ENDPOINT`` is set.

Throughput diagnostics
----------------------
``diagnose_throughput`` models expected decode throughput from the
serving config with a simple, documented heuristic (batch size,
KV-cache headroom, tensor parallelism, quantization, speculative
decoding) and compares it with a measured tok/s figure. It answers the
JD's question — "why is the GPU only hitting 3 tok/s when it should
do 18?" — with ranked likely causes. The model is a *heuristic for
triage*, not a measured GPU benchmark; the README says so.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from network_intelligence.config import ServingConfig

log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    tokens_per_s: float
    backend: str
    simulated: bool = False  # True when numbers come from the mock model

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(self.latency_s, 3),
            "tokens_per_s": round(self.tokens_per_s, 2),
            "backend": self.backend,
            "simulated": self.simulated,
        }


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockBackend:
    """Deterministic CPU stand-in for an LLM serving engine.

    ``generate`` returns a templated completion whose *simulated*
    latency is computed from the serving config (see
    :func:`model_tokens_per_s`), so the throughput diagnostic has
    something honest to chew on. ``plan`` is the agent's planner: a
    rule-based stand-in for a fine-tuned diagnostic model. It picks the
    next tool from the incident's signals and the investigation history.
    Swap it for a real model by changing ``ServingConfig.backend``.
    """

    # Ordered investigation plan: each step names the tool and how its
    # arguments are derived from the incident.
    PLAN = ["query_metrics", "search_logs", "check_alerts", "correlate_events", "decision_tree"]

    def __init__(self, cfg: ServingConfig) -> None:
        self.cfg = cfg
        self.backend_name = "mock"

    def generate(self, prompt: str, max_tokens: int | None = None) -> GenerationResult:
        started = time.monotonic()
        n_tokens = min(max_tokens or self.cfg.max_tokens, 64)
        text = (
            "[MOCK BACKEND — simulated output, not a real model] "
            "Diagnostic hypothesis: check KV-cache headroom, batch size, "
            "and recent interface errors before concluding. "
            f"Prompt received ({_approx_tokens(prompt)} tokens)."
        )
        # Simulated decode time from the throughput model, not wall clock.
        sim_tok_s = model_tokens_per_s(self.cfg)
        latency = n_tokens / max(sim_tok_s, 0.1)
        return GenerationResult(
            text=text,
            prompt_tokens=_approx_tokens(prompt),
            completion_tokens=n_tokens,
            latency_s=round(latency, 3),
            tokens_per_s=round(sim_tok_s, 2),
            backend=self.backend_name,
            simulated=True,
        )

    def plan(self, incident: dict, history: list[dict], context: list[dict]) -> dict:
        """Return the next action: {"tool": name, "args": {...}} or
        {"answer": True} once the plan is exhausted."""
        done = {h["tool"] for h in history}
        for tool in self.PLAN:
            if tool not in done:
                return {"tool": tool, "args": self._args_for(tool, incident)}
        return {"answer": True}

    def _args_for(self, tool: str, incident: dict) -> dict:
        device = incident.get("device", "")
        signals = incident.get("signals", {})
        if tool == "query_metrics":
            metrics = {
                "kv_cache_exhaustion": ["inference_tok_per_s", "kv_cache_usage_pct", "p99_latency_ms"],
                "faulty_optic": ["packet_loss_pct", "interface_errors"],
                "batching_too_small": ["gpu_util_pct", "inference_tok_per_s"],
            }.get(incident.get("expected_root_cause", ""), list(signals))
            return {"device": device, "metrics": metrics or list(signals), "minutes": 30}
        if tool == "search_logs":
            return {"device": device, "level": "ERROR", "limit": 20}
        if tool == "check_alerts":
            return {"device": device}
        if tool == "correlate_events":
            return {"device": device, "minutes": 30}
        if tool == "decision_tree":
            return {"device": device, "signals": signals}
        return {"device": device}


class VLLMBackend:
    """Real vLLM engine. GPU only; raises a clear error without vLLM."""

    def __init__(self, cfg: ServingConfig) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install the GPU stack "
                "(see README 'Running on a real GPU') or use backend='mock'."
            ) from exc
        self._llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            quantization=cfg.quantization if cfg.quantization != "none" else None,
            max_num_seqs=cfg.max_num_seqs,
            gpu_memory_utilization=0.9,
        )
        self._sampling_cls = SamplingParams
        self.cfg = cfg
        self.backend_name = "vllm"

    def generate(self, prompt: str, max_tokens: int | None = None) -> GenerationResult:
        started = time.monotonic()
        params = self._sampling_cls(
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        out = self._llm.generate([prompt], params)[0]
        latency = time.monotonic() - started
        n = len(out.outputs[0].token_ids)
        return GenerationResult(
            text=out.outputs[0].text,
            prompt_tokens=len(out.prompt_token_ids),
            completion_tokens=n,
            latency_s=round(latency, 3),
            tokens_per_s=round(n / max(latency, 1e-6), 2),
            backend=self.backend_name,
            simulated=False,
        )

    def plan(self, incident: dict, history: list[dict], context: list[dict]) -> dict:
        raise NotImplementedError(
            "vLLM planning needs a fine-tuned tool-calling model; "
            "use the mock planner for the demo or wire your own prompt."
        )


class OpenAICompatibleBackend:
    """Posts to an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, cfg: ServingConfig) -> None:
        if not cfg.endpoint:
            raise RuntimeError("Set NIA_LLM_ENDPOINT for backend='openai-compatible'.")
        self.cfg = cfg
        self.backend_name = "openai-compatible"

    def generate(self, prompt: str, max_tokens: int | None = None) -> GenerationResult:
        import httpx

        started = time.monotonic()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.cfg.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(self.cfg.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        latency = time.monotonic() - started
        choice = data["choices"][0]
        usage = data.get("usage", {})
        n = usage.get("completion_tokens", _approx_tokens(choice["message"]["content"]))
        return GenerationResult(
            text=choice["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", _approx_tokens(prompt)),
            completion_tokens=n,
            latency_s=round(latency, 3),
            tokens_per_s=round(n / max(latency, 1e-6), 2),
            backend=self.backend_name,
            simulated=False,
        )

    def plan(self, incident: dict, history: list[dict], context: list[dict]) -> dict:
        raise NotImplementedError("Tool-calling via the generic endpoint is not wired in the demo.")


def get_backend(cfg: ServingConfig):
    if cfg.backend == "mock":
        return MockBackend(cfg)
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    if cfg.backend == "openai-compatible":
        return OpenAICompatibleBackend(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend!r} (mock | vllm | openai-compatible)")


# ---------------------------------------------------------------------------
# Throughput model + diagnostics
# ---------------------------------------------------------------------------

# Baseline decode rate for a ~8B model on one modern datacentre GPU at
# healthy batching, in tok/s. This is an *assumption* for the heuristic
# model, stated here so it can be challenged — not a measured benchmark.
BASE_DECODE_TOK_S = 18.0


def model_tokens_per_s(cfg: ServingConfig, kv_cache_usage_pct: float = 55.0) -> float:
    """Heuristic expected decode throughput (tok/s) for a serving config.

    Factors (multiplicative, each documented):
    * batch: throughput scales with batch up to ~32, then saturates.
    * KV cache: no penalty up to 80% usage; above that the cache forces
      ever-smaller batches, down to 5% of peak at 100% usage.
    * tensor parallel: ~0.85x efficiency per added GPU (communication).
    * quantization: fp8/awq/gptq roughly 1.3x decode (less memory traffic).
    * speculative decoding: 1 / (1 - accept_rate * 0.5) up to ~1.6x.
    """
    batch = max(1, cfg.max_num_seqs)
    batch_factor = min(1.0, batch / 32.0)
    if kv_cache_usage_pct <= 80:
        kv_factor = 1.0
    else:
        kv_factor = max(0.05, 1.0 - (kv_cache_usage_pct - 80) / 20.0 * 0.95)
    tp_factor = 1.0 + 0.85 * (cfg.tensor_parallel_size - 1) if cfg.tensor_parallel_size > 1 else 1.0
    quant_factor = {"fp8": 1.3, "awq": 1.25, "gptq": 1.2}.get(cfg.quantization, 1.0)
    spec_factor = 1.0
    if cfg.enable_speculative:
        spec_factor = min(1.6, 1.0 / max(0.2, 1.0 - cfg.speculative_accept_rate * 0.5))
    return BASE_DECODE_TOK_S * batch_factor * tp_factor * quant_factor * spec_factor * kv_factor


@dataclass
class ThroughputDiagnosis:
    measured_tok_s: float
    expected_tok_s: float
    ratio: float  # measured / expected
    likely_causes: list[dict] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "measured_tok_s": self.measured_tok_s,
            "expected_tok_s": round(self.expected_tok_s, 2),
            "ratio": round(self.ratio, 3),
            "likely_causes": self.likely_causes,
            "recommendation": self.recommendation,
        }


def diagnose_throughput(
    cfg: ServingConfig,
    measured_tok_s: float,
    kv_cache_usage_pct: float = 55.0,
    batch_size: int | None = None,
    speculative_accept_rate: float | None = None,
) -> ThroughputDiagnosis:
    """Explain a throughput gap: measured vs the heuristic expectation.

    The expectation is computed for a *healthy* cache (55% usage): it
    is what this config should deliver. The observed cache pressure is
    then offered as the explanation for the gap. All causes are
    hypotheses to verify against metrics/logs — the agent's job — not
    conclusions.
    """
    expected = model_tokens_per_s(cfg)
    ratio = measured_tok_s / expected if expected else 0.0
    causes: list[dict] = []
    if kv_cache_usage_pct >= 90:
        causes.append({
            "cause": "kv_cache_exhaustion",
            "detail": f"KV cache at {kv_cache_usage_pct}% — only tiny batches fit, decode collapses.",
            "check": "query_metrics kv_cache_usage_pct; search_logs for 'KV cache allocation failed'",
            "confidence": 0.9,
        })
    eff_batch = batch_size if batch_size is not None else cfg.max_num_seqs
    if eff_batch is not None and eff_batch < 8:
        causes.append({
            "cause": "batching_too_small",
            "detail": f"Running batch ~{eff_batch}: decode is memory-bandwidth bound, small batches waste the GPU.",
            "check": "query_metrics gpu_util_pct + inference_tok_per_s; raise max_num_seqs",
            "confidence": 0.8,
        })
    if cfg.enable_speculative and (speculative_accept_rate or cfg.speculative_accept_rate) < 0.5:
        causes.append({
            "cause": "speculative_decoding_low_acceptance",
            "detail": "Draft model acceptance is low — most speculative work is discarded.",
            "check": "compare draft vs target model outputs on recent traces",
            "confidence": 0.7,
        })
    if cfg.tensor_parallel_size > 1 and ratio < 0.5 and kv_cache_usage_pct < 90:
        causes.append({
            "cause": "tensor_parallel_overhead",
            "detail": "Tensor parallelism adds all-reduce communication per layer; small models may be slower sharded.",
            "check": "try tensor_parallel_size=1 for this model size",
            "confidence": 0.5,
        })
    if not causes:
        causes.append({
            "cause": "no_config_cause_found",
            "detail": "Serving config looks healthy — look at the model, the network path, or a bad deploy.",
            "check": "correlate_events across serving + network devices",
            "confidence": 0.4,
        })
    causes.sort(key=lambda c: c["confidence"], reverse=True)
    rec = (
        f"Most likely: {causes[0]['cause']}. {causes[0]['check']}."
        if causes else "No cause identified."
    )
    return ThroughputDiagnosis(
        measured_tok_s=measured_tok_s,
        expected_tok_s=expected,
        ratio=ratio,
        likely_causes=causes,
        recommendation=rec,
    )
