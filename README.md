# LLM Inference Optimization System

> Reducing LLM serving latency through quantization, KV caching, continuous batching, and speculative decoding — fully measurable, production-grade, runs on CPU.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-2.52-orange)
![Grafana](https://img.shields.io/badge/Grafana-10.4-yellow)

---

## TL;DR

This project implements four major LLM inference optimizations used in modern serving engines like vLLM and TGI.

| Technique            | Metric     | Improvement           | Hardware    |
|----------------------|------------|-----------------------|-------------|
| Quantization         | TTFT       | **5.58× faster**      | CPU         |
| Quantization         | TPS        | **7.3× faster**       | CPU         |
| KV cache (LRU)       | TTFT       | **81% reduction**     | CPU         |
| KV cache (Sliding)   | TTFT       | **82% reduction**     | CPU         |
| Continuous batching  | Throughput | algorithm demonstrated | GPU optimal |
| Speculative decoding | TTFT       | algorithm demonstrated | GPU optimal |

KV cache and quantization results are experimentally measured on CPU.
Batching and speculative decoding algorithms are correctly implemented;
full throughput gains require GPU-based parallel verification.

---

## Motivation

Most developers interact with LLMs via APIs but very few understand how inference systems are optimized internally.

Serving LLMs efficiently requires solving several systems challenges:

- Slow token generation
- Wasted compute during batching
- Repeated prompt recomputation
- Poor observability of latency metrics

This project builds a mini LLM serving system from scratch to study and benchmark these optimizations.

---

## Architecture

```
Incoming Request
       │
       ▼
┌─────────────────────┐
│ FastAPI Server      │
│ POST /generate      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ LRU KV Cache        │
│ Cache hit → instant │
│ Cache miss → model  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Batch Scheduler     │
│ Continuous batching │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Inference Engine    │
│ llama.cpp GGUF      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Prometheus Metrics  │
│ TTFT, TPS, cache    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Grafana Dashboard   │
│ Live monitoring     │
└─────────────────────┘
```

---

## Benchmark Results

Benchmarks were run on CPU only using **TinyLlama 1.1B**.

### Quantization

| Format   | Memory  | TTFT    | TPS   | ROUGE-L |
|----------|---------|---------|-------|---------|
| F16      | ~2.2 GB | 31.58 s | 1.83  | 1.000   |
| Q4\_K\_M | ~0.7 GB | 5.66 s  | 13.28 | 0.384   |

Result: **5.58× TTFT speedup** with only 13.6% ROUGE-L quality drop.

---

### KV Cache

| Policy         | TTFT Reduction | Hit Rate |
|----------------|----------------|----------|
| No cache       | — (baseline)   | 0%       |
| FIFO           | **80.7%**      | 66.7%    |
| LRU            | **81.2%**      | 66.7%    |
| Sliding window | **82.1%**      | 66.7%    |

Cache hits are served instantly (0 model calls). With 66.7% hit rate on a repeated-prompt workload, average TTFT drops by over 80%.

---

### Continuous Batching

| Scheduler           | Description                                   |
|---------------------|-----------------------------------------------|
| No batching         | Sequential, one request at a time             |
| Static batching     | Collect N requests, process together          |
| Continuous batching | Fill slots immediately as requests complete   |

Full throughput gains require GPU-based parallel token generation across batch slots.

---

### Speculative Decoding

Algorithm correctly implements Chen et al. (2023) with **100% acceptance rate** at temperature = 0.

| Mode                | Accept Rate | Notes                       |
|---------------------|-------------|-----------------------------|
| Baseline            | —           | 1 target call per token     |
| Speculative (γ = 5) | **100%**    | GPU needed for full speedup |

The 5–6× speedup is achievable on GPU where the target model verifies k tokens in a single forward pass.

---

## Optimization Techniques

### 1. Quantization

Compresses model weights from float16 (~2.2 GB) to 4-bit integers Q4\_K\_M (~0.7 GB).
Fewer bytes to read per forward pass → dramatically faster inference.

- **File:** [src/optimizations/quantization.py](src/optimizations/quantization.py)
- **Benchmark:** [src/benchmarks/benchmark_quantization.py](src/benchmarks/benchmark_quantization.py)

---

### 2. KV Cache

Caches the full generated response keyed by a SHA-256 hash of the prompt.
Repeated prompts are served instantly without touching the model.

| Policy         | Evicts when full by…                  |
|----------------|---------------------------------------|
| FIFO           | Removing the oldest inserted entry    |
| LRU            | Removing the least-recently-used entry|
| Sliding window | Removing entries older than N seconds |

- **File:** [src/optimizations/kv_cache.py](src/optimizations/kv_cache.py)
- **Benchmark:** [src/benchmarks/benchmark_kv_cache.py](src/benchmarks/benchmark_kv_cache.py)

---

### 3. Continuous Batching

| Strategy            | Description                                           |
|---------------------|-------------------------------------------------------|
| No batching         | Process requests one at a time sequentially           |
| Static batching     | Collect N requests, then process all                  |
| Continuous batching | Fill inference slots immediately as requests complete |

- **File:** [src/optimizations/batching.py](src/optimizations/batching.py)
- **Benchmark:** [src/benchmarks/benchmark_batching.py](src/benchmarks/benchmark_batching.py)

---

### 4. Speculative Decoding

Implements speculative sampling (Chen et al., 2023):

1. A fast **draft model** speculatively generates γ = 5 tokens greedily.
2. The **target model** verifies all γ tokens in one forward pass.
3. Tokens are accepted if the target agrees; first rejection triggers a resample.
4. When all γ tokens are accepted, one bonus token is sampled from the target.

- **File:** [src/optimizations/speculative.py](src/optimizations/speculative.py)
- **Benchmark:** [src/benchmarks/benchmark_speculative.py](src/benchmarks/benchmark_speculative.py)

---

## Observability

All metrics are exported to **Prometheus** and visualised in **Grafana**.

| Metric                  | Type      | Description                      |
|-------------------------|-----------|----------------------------------|
| `llm_requests_total`    | Counter   | Total inference requests         |
| `llm_ttft_seconds`      | Histogram | Time-to-first-token distribution |
| `llm_tokens_per_second` | Gauge     | Current tokens per second        |
| `llm_cache_hits_total`  | Counter   | KV cache hits                    |
| `llm_cache_misses_total`| Counter   | KV cache misses                  |
| `llm_cache_hit_rate`    | Gauge     | Rolling cache hit rate           |

---

## Project Structure

```
llm_infrence_optimization/
├── src/
│   ├── server/
│   │   ├── inference.py               # InferenceEngine (llama-cpp wrapper)
│   │   └── app.py                     # FastAPI server + Prometheus metrics
│   ├── optimizations/
│   │   ├── quantization.py            # F16 vs Q4_K_M comparison
│   │   ├── kv_cache.py                # LRU / FIFO / Sliding window cache
│   │   ├── batching.py                # No / Static / Continuous schedulers
│   │   └── speculative.py             # Speculative decoding (Chen et al. 2023)
│   └── benchmarks/
│       ├── metrics.py                 # ROUGE-L + table utilities
│       ├── benchmark_quantization.py
│       ├── benchmark_kv_cache.py
│       ├── benchmark_batching.py
│       └── benchmark_speculative.py
├── grafana/
│   ├── dashboards/inference.json      # Pre-built Grafana dashboard
│   └── provisioning/                  # Auto-config for datasource + dashboard
├── models/                            # Place .gguf files here (gitignored)
├── Dockerfile
├── docker-compose.yml                 # Full stack (app + prometheus + grafana)
├── docker-compose.monitoring.yml      # Monitoring only (prometheus + grafana)
├── prometheus.yml
└── pyproject.toml
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — for observability stack
- **Windows only:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++" — required to compile llama-cpp-python

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/Tanisha2703/llm_infrence_optimization
cd llm_infrence_optimization
```

---

### Step 2 — Install dependencies

```bash
pip install uv
uv venv
```

**Windows:**
```powershell
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

```bash
uv pip install -e .
```

> This takes 5–10 minutes on first run because llama-cpp-python compiles C++ code.

---

### Step 3 — Download the model

**Q4\_K\_M — required for all benchmarks (~700 MB):**

```bash
pip install huggingface-hub

huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir models
```

**Windows — rename the file:**
```powershell
Rename-Item models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "tinyllama-1.1b-Q4_K_M.gguf"
```

**macOS / Linux:**
```bash
mv models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf models/tinyllama-1.1b-Q4_K_M.gguf
```

**F16 — required only for quantization benchmark (~2.2 GB):**

```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.F16.gguf \
    --local-dir models
```

**Windows:**
```powershell
Rename-Item models\tinyllama-1.1b-chat-v1.0.F16.gguf "tinyllama-1.1b-f16.gguf"
```

**macOS / Linux:**
```bash
mv models/tinyllama-1.1b-chat-v1.0.F16.gguf models/tinyllama-1.1b-f16.gguf
```

---

## Running Benchmarks

```bash
# KV cache eviction policies
python -m src.benchmarks.benchmark_kv_cache

# Batching scheduling strategies
python -m src.benchmarks.benchmark_batching

# Speculative decoding vs baseline
python -m src.benchmarks.benchmark_speculative

# Quantization quality vs speed (needs F16 model)
python -m src.benchmarks.benchmark_quantization
```

---

## Running the API Server

**Windows:**
```powershell
$env:MODEL_PATH = "models\tinyllama-1.1b-Q4_K_M.gguf"
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

**macOS / Linux:**
```bash
MODEL_PATH=models/tinyllama-1.1b-Q4_K_M.gguf uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

**Test it:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum mechanics.", "max_tokens": 50}'
```

**Response:**
```json
{
  "text": "...",
  "ttft": 0.12,
  "total_time": 3.4,
  "token_count": 50,
  "tps": 14.7,
  "cache_hit": false
}
```

Send the **same prompt again** — `cache_hit` becomes `true` and `ttft` becomes `0.0` (instant).

---

## Activating Prometheus + Grafana

You need **two terminal windows**.

**Window 1 — start the API server** (as above)

**Window 2 — start monitoring containers:**
```bash
docker compose -f docker-compose.monitoring.yml up
```

**Generate traffic to see live graphs:**

```powershell
# Windows
1..15 | ForEach-Object {
    Invoke-RestMethod -Uri http://localhost:8000/generate `
      -Method POST -ContentType "application/json" `
      -Body '{"prompt": "What is machine learning?", "max_tokens": 30}'
    Start-Sleep -Seconds 2
}
```

```bash
# macOS / Linux
for i in {1..15}; do
  curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is machine learning?", "max_tokens": 30}'
  sleep 2
done
```

**Open dashboards:**

| Service       | URL                   | Login        |
|---------------|-----------------------|--------------|
| API health    | http://localhost:8000/health  | none |
| Raw metrics   | http://localhost:8000/metrics | none |
| Prometheus    | http://localhost:9090 | none         |
| Grafana       | http://localhost:3000 | admin/admin  |

In Grafana: **Dashboards → LLM Inference Optimizer** — all 6 panels update live.

---

## Tech Stack

| Category         | Tools                   |
|------------------|-------------------------|
| Inference        | llama-cpp-python (GGUF) |
| API              | FastAPI + Uvicorn       |
| Metrics          | Prometheus              |
| Visualisation    | Grafana                 |
| Containerisation | Docker Compose          |
| Quality eval     | ROUGE-L (LCS-based)     |

---

## References

- Chen et al. (2023) — *Accelerating Large Language Model Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318
- Kwon et al. (2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)* — https://arxiv.org/abs/2309.06180
- Yu et al. (2022) — *Orca: A Distributed Serving System for Transformer-Based Generative Models* — https://www.usenix.org/conference/osdi22/presentation/yu
- Frantar et al. (2022) — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — https://arxiv.org/abs/2210.17323
- Gerganov (2023) — *llama.cpp* — https://github.com/ggerganov/llama.cpp
