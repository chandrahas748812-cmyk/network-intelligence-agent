"""Serving layer tests: mock backend behaviour and the throughput
diagnostic — the 'why is the GPU only hitting 3 tok/s?' check."""

import pytest

from network_intelligence.config import ServingConfig
from network_intelligence.serving import (
    MockBackend,
    VLLMBackend,
    diagnose_throughput,
    get_backend,
    model_tokens_per_s,
)


def test_mock_generate_is_labelled_simulated():
    result = MockBackend(ServingConfig()).generate("diagnose this incident")
    assert result.simulated is True
    assert result.backend == "mock"
    assert result.tokens_per_s > 0
    assert result.completion_tokens > 0


def test_model_tokens_per_s_scales_with_batch():
    healthy = model_tokens_per_s(ServingConfig(max_num_seqs=64))
    tiny_batch = model_tokens_per_s(ServingConfig(max_num_seqs=1))
    assert healthy == pytest.approx(18.0)
    assert tiny_batch < healthy
    assert tiny_batch == pytest.approx(18.0 / 32)


def test_model_tokens_per_s_collapses_with_full_kv_cache():
    collapsed = model_tokens_per_s(ServingConfig(max_num_seqs=64), kv_cache_usage_pct=97.0)
    assert collapsed == pytest.approx(18.0 * 0.1925, abs=0.01)


def test_model_tokens_per_s_quantization_and_speculative_help():
    base = model_tokens_per_s(ServingConfig(max_num_seqs=64))
    fp8 = model_tokens_per_s(ServingConfig(max_num_seqs=64, quantization="fp8"))
    spec = model_tokens_per_s(
        ServingConfig(max_num_seqs=64, enable_speculative=True, speculative_accept_rate=0.8)
    )
    assert fp8 > base
    assert spec > base


def test_diagnose_throughput_kv_cache_exhaustion():
    diag = diagnose_throughput(
        ServingConfig(max_num_seqs=64), measured_tok_s=3.1,
        kv_cache_usage_pct=97.0, batch_size=1,
    )
    assert diag.likely_causes[0]["cause"] == "kv_cache_exhaustion"
    assert diag.ratio < 0.5
    assert "kv_cache" in diag.recommendation


def test_diagnose_throughput_small_batch():
    diag = diagnose_throughput(
        ServingConfig(max_num_seqs=8), measured_tok_s=6.5,
        kv_cache_usage_pct=55.0, batch_size=4,
    )
    causes = [c["cause"] for c in diag.likely_causes]
    assert "batching_too_small" in causes


def test_diagnose_throughput_healthy_config_says_so():
    diag = diagnose_throughput(
        ServingConfig(max_num_seqs=64), measured_tok_s=17.5, kv_cache_usage_pct=55.0
    )
    assert diag.ratio > 0.9
    assert diag.likely_causes[0]["cause"] == "no_config_cause_found"


def test_vllm_backend_errors_clearly_without_vllm():
    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        VLLMBackend(ServingConfig(backend="vllm"))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend(ServingConfig(backend="trtllm"))
