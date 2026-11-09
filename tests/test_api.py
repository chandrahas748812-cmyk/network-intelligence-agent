"""API tests: the FastAPI layer end to end via TestClient (no live server)."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_labels_mock_backend():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"
    assert body["simulated"] is True


def test_generate_is_simulated():
    resp = client.post("/generate", json={"prompt": "diagnose this incident"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["simulated"] is True
    assert body["tokens_per_s"] > 0
    assert "MOCK BACKEND" in body["text"]


def test_investigate_endpoint():
    resp = client.post("/investigate", json={
        "id": "INC-1001",
        "title": "LLM inference throughput collapsed to ~3 tok/s",
        "device": "gpu-node-02",
        "severity": "critical",
        "signals": {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0,
                    "p99_latency_ms": 4200.0, "gpu_util_pct": 34.0},
        "expected_root_cause": "kv_cache_exhaustion",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["root_cause"] == "kv_cache_exhaustion"
    assert body["evidence"]
    assert body["playbook_id"] == "pb-throughput-collapse"


def test_watcher_scan_endpoint():
    resp = client.post("/watcher/scan")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    assert any(a["metric"] == "kv_cache_usage_pct" for a in body["alerts"])


def test_eval_endpoint():
    resp = client.get("/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert body["root_cause_accuracy"] == 1.0
    assert body["false_alarm_rate"] == 0.0
    assert body["n_cases"] == 4


def test_throughput_diagnose_endpoint():
    resp = client.post("/throughput/diagnose", json={
        "measured_tok_s": 3.1, "kv_cache_usage_pct": 97.0, "batch_size": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["likely_causes"][0]["cause"] == "kv_cache_exhaustion"
    assert body["ratio"] < 0.5


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    versions = {v["version"] for v in resp.json()["versions"]}
    assert "v0.1.0" in versions
    assert "v0.2.0-lora-sft" in versions
