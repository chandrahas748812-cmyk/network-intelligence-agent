"""Fine-tuning curation + store fallback tests."""

from network_intelligence.finetune import (
    curate_dataset,
    lora_config,
    qlora_config,
    trace_to_example,
)
from network_intelligence.stores import DedupCache, PostgresMetricsStore


def test_trace_to_example_shape():
    trace = {
        "incident": {"id": "INC-1", "title": "t", "device": "d", "signals": {}},
        "steps": [{"tool": "query_metrics", "args": {}, "observation": "tok/s 3.1"}],
        "diagnosis": {"root_cause": "kv_cache_exhaustion", "evidence": ["e1"],
                      "recommended_action": "free cache"},
    }
    ex = trace_to_example(trace)
    assert "kv_cache_exhaustion" in ex.output
    assert "query_metrics" in ex.input
    assert ex.instruction.startswith("You are a network-intelligence")


def test_curate_dataset(tmp_path):
    out = tmp_path / "sft.jsonl"
    stats = curate_dataset("data/traces.jsonl", out)
    assert stats["traces_in"] == 6
    assert stats["examples_out"] == 6
    assert stats["dropped"] == 0
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 6
    import json

    first = json.loads(lines[0])
    assert set(first) == {"instruction", "input", "output"}


def test_lora_configs_shape():
    lora = lora_config()
    assert lora["peft"]["r"] == 16
    assert "q_proj" in lora["peft"]["target_modules"]
    qlora = qlora_config()
    assert qlora["quantization"]["load_in_4bit"] is True
    assert qlora["quantization"]["bnb_4bit_quant_type"] == "nf4"


def test_dedup_cache_falls_back_without_redis():
    cache = DedupCache("redis://localhost:59999/0")  # nothing listening
    cache.set("k", {"v": 1}, ttl_s=60)
    assert cache.get("k") == {"v": 1}
    assert cache.seen("once") is False
    assert cache.seen("once") is True  # second time: already seen


def test_postgres_store_falls_back_without_server():
    store = PostgresMetricsStore("postgresql://nia:nia@localhost:59999/nia")
    assert store.live is False
    store.points = [{"ts": "t", "device": "d", "metric": "m", "value": 3.0}]
    assert store.latest_value("d", "m") == 3.0  # in-memory fallback answers
