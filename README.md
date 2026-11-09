# Network Intelligence Agent

An investigation agent for LLM-inference infrastructure: it watches
telemetry, investigates incidents by calling tools step by step,
retrieves runbooks and past incidents, and writes structured diagnostic
reports — plus the serving, fine-tuning, eval, and infra pieces around
it.

> **Portfolio / learning project.** Built to demonstrate the skills in
> the Tesla **Sr. ML Engineer, Network Intelligence** posting (Fremont,
> Req 284510): production LLM serving, agent/tool-calling loops,
> investigation playbooks, retrieval over runbooks, fine-tuning on
> investigation traces, evals that measure diagnosis correctness, and
> GPU infrastructure. It is **not employer work**, contains no
> proprietary data, and all telemetry, incidents, runbooks, and traces
> are **synthetic**. It makes **no claims about real Tesla systems**,
> and no GPU serving numbers here are measured — the mock backend is
> labelled as a stand-in wherever it appears.

## Architecture

```mermaid
flowchart TB
    subgraph Telemetry
        GEN[Synthetic generator<br/>telemetry.py]
        PG[(Postgres<br/>metrics / logs / alerts)]
        RD[(Redis<br/>dedup cache)]
        MEM[(In-memory stores<br/>demo default)]
    end
    subgraph Watcher
        W[TelemetryWatcher<br/>z-score + thresholds]
    end
    subgraph Agent
        A[InvestigationAgent<br/>agent.py]
        P[Mock planner<br/>or real model]
        T[Tools: metrics / logs / alerts<br/>threshold / decision tree<br/>correlation / runbook]
        R[BM25 retriever<br/>runbooks + past incidents]
    end
    subgraph Serving
        API[FastAPI<br/>/generate /investigate<br/>/watcher/scan /eval]
        BE[Backend: mock | vLLM<br/>| OpenAI-compatible]
        TD[Throughput diagnostic<br/>why 3 tok/s?]
    end
    subgraph Learning
        FT[Trace -> JSONL curation<br/>LoRA / QLoRA configs]
        EV[Eval harness<br/>golden cases]
        REG[Model registry<br/>eval-gated promotion]
    end
    GEN --> PG
    GEN --> MEM
    PG --> W
    MEM --> W
    W -->|candidate incident| A
    A --> P
    P --> T
    T --> PG
    T --> MEM
    A --> R
    API --> BE
    API --> A
    API --> TD
    A -->|trace| FT
    FT --> EV
    EV --> REG
```

## How each module maps to the Tesla JD

| JD requirement | Where it lives |
|---|---|
| LLM inference fast and reliable in production (quantization, tensor parallelism, batching, KV-cache tuning, speculative decoding) | `network_intelligence/serving.py` — `ServingConfig` + `model_tokens_per_s` heuristic; `configs/serving.yaml` |
| Debugging GPU throughput ("why 3 tok/s when it should do 18?") | `diagnose_throughput` in `serving.py`; `POST /throughput/diagnose` |
| Agent loop: model calls tools step by step, writes a diagnostic report; retries, timeouts, dedup, context limits | `agent.py` + `tools.py` (`ToolRunner`) |
| Investigation tools / playbooks: metrics queries, log search, alert checks, threshold comparisons, decision trees, event correlation | `tools.py`, `playbooks.py`, `data/runbooks.json` |
| Fine-tune on real investigation traces; evals that measure whether the diagnosis is actually correct | `finetune.py` + `train_sft.py` + `configs/lora_sft.yaml` / `qlora.yaml`; `evals.py` + `data/golden_cases.json` |
| Background agents watching telemetry continuously | `watcher.py` |
| Retrieval over runbooks and past incidents (RAG) | `retrieval.py` (BM25, CPU-only) + `data/runbooks.json`, `data/incidents.json` |
| GPU infra: multi-node clusters, Ray, Docker, K8s, model versioning, 1 vs 4 GPUs | `network_intelligence/ray_serve.py`, `configs/ray_serve.yaml`, `Dockerfile`, `docker-compose.yml`, `k8s/`, `registry.py` |
| Solid Python: FastAPI, PostgreSQL, Redis, Docker, Kubernetes | `api/main.py`, `stores.py`, `Dockerfile`, `k8s/` |

## Quickstart (no Docker, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py          # seeds data, runs watcher + one investigation + evals
pytest -q               # 52 tests
python -m network_intelligence.evals   # golden-case evals only
uvicorn api.main:app --port 8000       # API: /health /generate /investigate /eval
```

The demo runs entirely on CPU with the mock backend. It prints a
watcher pass, a full investigation of the throughput-collapse incident
(the JD's 3 tok/s case), the throughput diagnostic, one curated
fine-tuning example, and the eval scores.

### With Docker (real Postgres + Redis)

```bash
docker compose up --build
# API on :8000; stores switch automatically via DATABASE_URL / REDIS_URL.
# Without Docker the same code falls back to in-memory stores.
```

## What the demo shows

```
[2] Watcher pass: 10 new anomaly flag(s).
    - kv_cache_usage_pct on gpu-node-02 = 96.48 (threshold 90.0) -> pb-throughput-collapse
    - inference_tok_per_s on gpu-node-02 = 2.98 (z=-11.5) -> pb-throughput-collapse
    - packet_loss_pct on leaf-01 = 2.26 (threshold 1.0) -> pb-packet-loss
    ...
[3] Investigating INC-1001: LLM inference throughput collapsed to ~3 tok/s
Root cause: kv_cache_exhaustion (confidence 0.90)
...
[6] Eval harness (golden cases):
Eval: 4 golden cases
  root-cause accuracy: 100%
  mean evidence score: 1.00
  false-alarm rate:    0%
```

## API

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness + backend label (`simulated: true` for the mock) |
| `POST /generate` | LLM completion; returns tok/s and `simulated` flag |
| `POST /investigate` | Full agent investigation -> diagnostic report JSON |
| `POST /watcher/scan` | One watcher pass over telemetry |
| `GET /eval` | Golden-case eval scores |
| `GET /models` | Model registry versions + eval gates |
| `POST /throughput/diagnose` | Ranked likely causes for a measured tok/s gap |

Example:

```bash
curl -X POST localhost:8000/investigate -H 'Content-Type: application/json' -d '{
  "id": "INC-1001", "title": "LLM inference throughput collapsed to ~3 tok/s",
  "device": "gpu-node-02", "severity": "critical",
  "signals": {"inference_tok_per_s": 3.1, "kv_cache_usage_pct": 97.0,
              "p99_latency_ms": 4200.0, "gpu_util_pct": 34.0},
  "expected_root_cause": "kv_cache_exhaustion"}'
```

## Fine-tuning on real traces

```bash
python -c "from network_intelligence.finetune import curate_dataset; \
  print(curate_dataset('data/traces.jsonl', 'data/sft_train.jsonl'))"
# -> 6 examples from 6 traces, standard {"instruction","input","output"} SFT shape

python train_sft.py --config configs/lora_sft.yaml   # GPU only; refuses without torch + CUDA
```

Configs: `configs/lora_sft.yaml` (LoRA r=16) and `configs/qlora.yaml`
(4-bit NF4 base). The registry (`data/model_registry.json`) only
promotes a version to production after a recorded eval clears the
0.75 root-cause-accuracy gate.

## Running on a real GPU

The CPU demo never claims GPU numbers. For a real run:

1. `pip install torch transformers peft datasets accelerate trl vllm "ray[serve]"`
2. Set `NIA_BACKEND=vllm` and `NIA_MODEL=<your model>` (or point
   `NIA_LLM_ENDPOINT` at a vLLM/TGI server with
   `NIA_BACKEND=openai-compatible`).
3. `uvicorn api.main:app` serves the real engine through the same
   `/generate` endpoint; `train_sft.py` runs the LoRA/QLoRA configs.
4. Ray Serve: `serve run configs/ray_serve.yaml`. One 8B model usually
   wants 1 GPU per replica + more replicas, not tensor-parallel
   sharding — measure with the throughput diagnostic before choosing.

## Honest limits

- The mock backend's latency and tok/s are **simulated from a
  documented heuristic** (`BASE_DECODE_TOK_S = 18`), not measured on a
  GPU. The throughput diagnostic is a triage heuristic, not a benchmark.
- The planner in the demo is rule-based (a stand-in for a fine-tuned
  tool-calling model). Swap it by setting `NIA_BACKEND` and wiring your
  model into `serving.py`.
- BM25 retrieval is lexical on purpose: no downloads, no GPU. A dense
  retriever (sentence-transformers) is a documented upgrade, not a
  hidden dependency.
- All devices, metrics, incidents, and runbooks are invented for the
  demo. Nothing here describes Tesla's real systems.
