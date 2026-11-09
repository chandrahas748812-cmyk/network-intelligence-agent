"""Configuration for the Network Intelligence Agent.

All settings are plain dataclasses with environment-variable overrides so
the same code runs in the local demo, in Docker Compose, and in
Kubernetes (via the ConfigMap in ``k8s/configmap.yaml``).

Nothing here is a secret. API keys, database passwords, and tokens are
never stored in this repo; connection strings come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ServingConfig:
    """LLM serving configuration, modelled on vLLM / TGI style options.

    The mock backend (``MockBackend``) honours these settings in its
    simulated throughput model; a real vLLM backend would pass them to
    the engine. Nothing here claims a measured GPU benchmark — the
    numbers the mock reports are labelled as simulated.
    """

    backend: str = "mock"  # "mock" | "vllm" | "openai-compatible"
    model: str = "mock-diagnostic-llm-0.1"
    # How many GPUs the model is sharded across (tensor parallelism).
    tensor_parallel_size: int = 1
    # Weight precision: "fp16" | "bf16" | "fp8" | "awq" | "gptq" | "none".
    quantization: str = "none"
    # Max concurrent sequences (batching). Small values under-utilise the GPU.
    max_num_seqs: int = 64
    # KV-cache memory budget in GiB. Too small => tiny batches => low tok/s.
    kv_cache_gb: float = 16.0
    # Speculative decoding: a small draft model proposes tokens.
    enable_speculative: bool = False
    draft_model: str = ""
    speculative_accept_rate: float = 0.7  # fraction of draft tokens kept
    # Sampling.
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    # OpenAI-compatible endpoint (only used when backend="openai-compatible").
    endpoint: str = ""
    api_key_env: str = "NIA_LLM_API_KEY"

    @classmethod
    def from_env(cls) -> "ServingConfig":
        return cls(
            backend=_env("NIA_BACKEND", "mock"),
            model=_env("NIA_MODEL", "mock-diagnostic-llm-0.1"),
            tensor_parallel_size=_env_int("NIA_TENSOR_PARALLEL_SIZE", 1),
            quantization=_env("NIA_QUANTIZATION", "none"),
            max_num_seqs=_env_int("NIA_MAX_NUM_SEQS", 64),
            kv_cache_gb=_env_float("NIA_KV_CACHE_GB", 16.0),
            enable_speculative=_env_bool("NIA_SPECULATIVE", False),
            draft_model=_env("NIA_DRAFT_MODEL", ""),
            max_tokens=_env_int("NIA_MAX_TOKENS", 512),
            temperature=_env_float("NIA_TEMPERATURE", 0.0),
            endpoint=_env("NIA_LLM_ENDPOINT", ""),
        )


@dataclass
class AgentConfig:
    """Investigation agent loop limits (the JD's retries / timeouts /
    dedup / context-limit requirements, made explicit)."""

    max_steps: int = 8
    max_retries: int = 3
    tool_timeout_s: float = 5.0
    # Approximate context budget in tokens (chars / 4 is the estimate).
    context_budget_tokens: int = 4096
    # Per-tool-output truncation, in characters.
    max_tool_output_chars: int = 2000
    # Dedup cache TTL for identical tool calls.
    dedup_ttl_s: int = 300

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            max_steps=_env_int("NIA_AGENT_MAX_STEPS", 8),
            max_retries=_env_int("NIA_AGENT_MAX_RETRIES", 3),
            tool_timeout_s=_env_float("NIA_AGENT_TOOL_TIMEOUT_S", 5.0),
            context_budget_tokens=_env_int("NIA_AGENT_CONTEXT_TOKENS", 4096),
            dedup_ttl_s=_env_int("NIA_AGENT_DEDUP_TTL_S", 300),
        )


@dataclass
class WatcherConfig:
    """Background watcher thresholds."""

    # Rolling z-score above which a metric value is flagged.
    z_threshold: float = 3.0
    # Minimum points in the rolling window before flagging.
    min_window: int = 10
    # Hard thresholds (metric name -> value) checked alongside z-scores.
    thresholds: dict = field(
        default_factory=lambda: {
            "packet_loss_pct": 1.0,
            "p99_latency_ms": 1000.0,
            "kv_cache_usage_pct": 90.0,
            "gpu_temp_c": 85.0,
        }
    )
    # How far back the watcher looks, in minutes.
    lookback_minutes: int = 30

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        return cls(
            z_threshold=_env_float("NIA_WATCHER_Z", 3.0),
            min_window=_env_int("NIA_WATCHER_MIN_WINDOW", 10),
        )


@dataclass
class StoreConfig:
    """Where metrics / incidents live. Defaults are in-memory so the
    demo runs with no infrastructure. Set the env vars to use the
    docker-compose Postgres / Redis services."""

    database_url: str = ""  # e.g. postgresql://nia:nia@localhost:5432/nia
    redis_url: str = ""  # e.g. redis://localhost:6379/0

    @classmethod
    def from_env(cls) -> "StoreConfig":
        return cls(
            database_url=_env("DATABASE_URL", ""),
            redis_url=_env("REDIS_URL", ""),
        )
