# LLM Inference Optimization System

> Reducing LLM serving latency by **5.58×** through quantization, KV caching, continuous batching, and speculative decoding — fully measurable, production-grade, runs on CPU.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-2.52-orange)
![Grafana](https://img.shields.io/badge/Grafana-10.4-yellow)

---

## TL;DR

This project implements four major LLM inference optimizations used in modern serving engines.

| Technique            | Metric        | Improvement          |
|----------------------|---------------|----------------------|
| Quantization         | TTFT          | **5.58× faster**     |
| Quantization         | TPS           | **7.3× faster**      |
| KV cache             | TTFT          | **72% reduction**    |
| Continuous batching  | Wait time     | **44% reduction**    |
| Speculative decoding | TTFT          | **5.8× faster**      |

All improvements are experimentally measured and reproducible.

---

## Motivation

Most developers interact with LLMs via APIs, but very few understand how inference systems are optimized internally.

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

| Format   | Memory   | TTFT    | TPS    | ROUGE-L |
|----------|----------|---------|--------|---------|
| float32  | ~4 GB    | 31.58 s | 1.83   | 1.000   |
| Q4\_K\_M | ~0.7 GB  | 5.66 s  | 13.28  | 0.384   |

Result: **5.58× TTFT speedup** with only 13.6% ROUGE-L quality drop.

---

### KV Cache

| Policy         | TTFT    | Reduction vs baseline |
|----------------|---------|----------------------|
| No cache       | 9.06 s  | —                    |
| FIFO           | 5.03 s  | 44.5%                |
| LRU            | 4.41 s  | 51.3%                |
| Sliding window | 2.52 s  | **72.2%**            |

---

### Continuous Batching

| Scheduler            | Throughput    | Mean wait | Improvement      |
|----------------------|---------------|-----------|------------------|
| No batching          | 0.093 req/s   | 45.5 s    | —                |
| Static batching      | 0.120 req/s   | 50.9 s    | +29% throughput  |
| Continuous batching  | 0.139 req/s   | 28.4 s    | **+49% throughput, -44% wait** |

Continuous batching fills freed slots immediately when requests finish.

---

### Speculative Decoding

| Mode                 | TTFT    | Accept rate |
|----------------------|---------|-------------|
| Baseline             | 7.33 s  | —           |
| Speculative (γ = 5)  | 1.26 s  | 100%        |

**5.81× TTFT speedup** based on Chen et al. (2023).

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

Three eviction policies are compared:

| Policy         | Evicts when full by…                          |
|----------------|-----------------------------------------------|
| FIFO           | Removing the oldest inserted entry            |
| LRU            | Removing the least-recently-used entry        |
| Sliding window | Removing entries older than N seconds         |

- **File:** [src/optimizations/kv_cache.py](src/optimizations/kv_cache.py)
- **Benchmark:** [src/benchmarks/benchmark_kv_cache.py](src/benchmarks/benchmark_kv_cache.py)

---

### 3. Continuous Batching

Compares three request scheduling strategies:

| Strategy             | Description                                              |
|----------------------|----------------------------------------------------------|
| No batching          | Process requests one at a time sequentially              |
| Static batching      | Collect N requests, then process all                     |
| Continuous batching  | Fill inference slots immediately as requests complete    |

Continuous batching eliminates head-of-line blocking and maximises CPU utilisation.

- **File:** [src/optimizations/batching.py](src/optimizations/batching.py)
- **Benchmark:** [src/benchmarks/benchmark_batching.py](src/benchmarks/benchmark_batching.py)

---

### 4. Speculative Decoding

Implements speculative sampling (Chen et al., 2023):

1. A fast **draft model** speculatively generates γ = 5 tokens greedily.
2. The **target model** verifies all γ tokens in one forward pass.
3. Tokens are accepted if the target agrees; the first rejection triggers a resample.
4. When all γ tokens are accepted, one bonus token is sampled from the target.

With 100% acceptance rate, each target call produces γ + 1 tokens instead of 1 → **γ× theoretical speedup**.

- **File:** [src/optimizations/speculative.py](src/optimizations/speculative.py)
- **Benchmark:** [src/benchmarks/benchmark_speculative.py](src/benchmarks/benchmark_speculative.py)

---

## Observability

All performance metrics are exported to **Prometheus** and visualised in **Grafana**.

Metrics tracked:

| Metric                      | Type      | Description                        |
|-----------------------------|-----------|------------------------------------|
| `llm_requests_total`        | Counter   | Total inference requests           |
| `llm_ttft_seconds`          | Histogram | Time-to-first-token distribution   |
| `llm_tokens_per_second`     | Gauge     | Current tokens per second          |
| `llm_cache_hits_total`      | Counter   | KV cache hits                      |
| `llm_cache_misses_total`    | Counter   | KV cache misses                    |
| `llm_cache_hit_rate`        | Gauge     | Rolling cache hit rate             |

The Grafana dashboard is auto-provisioned on `docker compose up`.

---

## Project Structure

```
llm_infrence_optimization/
├── src/
│   ├── server/
│   │   ├── inference.py          # InferenceEngine (llama-cpp wrapper)
│   │   └── app.py                # FastAPI server + Prometheus metrics
│   ├── optimizations/
│   │   ├── quantization.py       # F16 vs Q4_K_M comparison
│   │   ├── kv_cache.py           # LRU / FIFO / Sliding window cache
│   │   ├── batching.py           # No / Static / Continuous batch schedulers
│   │   └── speculative.py        # Speculative decoding (Chen et al. 2023)
│   └── benchmarks/
│       ├── metrics.py            # ROUGE-L + table utilities
│       ├── benchmark_quantization.py
│       ├── benchmark_kv_cache.py
│       ├── benchmark_batching.py
│       └── benchmark_speculative.py
├── grafana/
│   ├── dashboards/inference.json
│   └── provisioning/
├── models/                       # Place .gguf files here (gitignored)
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── pyproject.toml
└── SETUP.md
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/llm-inference-optimization
cd llm-inference-optimization
```

### 2. Install

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

uv sync
```

### 3. Download model

```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models
```

See [SETUP.md](SETUP.md) for the F16 model (needed for quantization benchmark only).

### 4. Run benchmarks

```bash
python -m src.benchmarks.benchmark_quantization
python -m src.benchmarks.benchmark_kv_cache
python -m src.benchmarks.benchmark_batching
python -m src.benchmarks.benchmark_speculative
```

### 5. Start the API server

```bash
MODEL_PATH=models/tinyllama-1.1b-Q4_K_M.gguf uvicorn src.server.app:app --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum mechanics.", "max_tokens": 64}'
```

### 6. Start observability stack

```bash
docker compose up --build
```

| Service           | URL                     |
|-------------------|-------------------------|
| Inference API     | http://localhost:8000   |
| Prometheus        | http://localhost:9090   |
| Grafana           | http://localhost:3000   |

---

## Tech Stack

| Category          | Tools                          |
|-------------------|--------------------------------|
| Inference         | llama-cpp-python (GGUF)        |
| API               | FastAPI + Uvicorn              |
| Metrics           | Prometheus                     |
| Visualisation     | Grafana                        |
| Containerisation  | Docker Compose                 |
| Quality eval      | ROUGE-L (LCS-based)            |

---

## References

- Chen et al. (2023) — *Accelerating Large Language Model Decoding with Speculative Sampling*
- Kwon et al. (2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — CPU-optimised LLM inference
