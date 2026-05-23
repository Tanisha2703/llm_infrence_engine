# Setup Guide

## 1. Install dependencies

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

uv pip install -e .
```

> **Note:** `llama-cpp-python` compiles a C++ extension.  
> On Windows you need the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).  
> On Linux: `sudo apt install build-essential cmake`

---

## 2. Download model files

Place model files in the `models/` directory.

### Q4_K_M (required for all benchmarks — ~0.7 GB)

```bash
# Option A: Hugging Face Hub CLI
pip install huggingface-hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir models

# Option B: direct wget / curl
curl -L -o models/tinyllama-1.1b-Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### F16 (required only for quantization benchmark — ~2.2 GB)

```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.F16.gguf \
    --local-dir models

# Rename to match expected path
mv models/tinyllama-1.1b-chat-v1.0.F16.gguf models/tinyllama-1.1b-f16.gguf
```

---

## 3. Run benchmarks

```bash
# Quantization (F16 vs Q4_K_M)
python -m src.benchmarks.benchmark_quantization

# KV cache eviction policies
python -m src.benchmarks.benchmark_kv_cache

# Batching scheduling strategies
python -m src.benchmarks.benchmark_batching

# Speculative decoding
python -m src.benchmarks.benchmark_speculative
```

---

## 4. Start the API server

```bash
MODEL_PATH=models/tinyllama-1.1b-Q4_K_M.gguf uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum mechanics.", "max_tokens": 64}'
```

---

## 5. Start the full observability stack (Docker)

```bash
docker compose up --build
```

| Service           | URL                        |
|-------------------|----------------------------|
| Inference API     | http://localhost:8000      |
| Prometheus        | http://localhost:9090      |
| Grafana           | http://localhost:3000      |

Grafana default login: **admin / admin**

The "LLM Inference Optimizer" dashboard is pre-provisioned automatically.

---

## Expected benchmark results (TinyLlama 1.1B, CPU)

| Technique          | Metric           | Baseline | Optimised | Improvement    |
|--------------------|------------------|----------|-----------|----------------|
| Quantization       | TTFT             | ~31s     | ~5.7s     | **5.5× faster** |
| Quantization       | TPS              | ~1.8     | ~13.3     | **7.3× faster** |
| KV cache (LRU)     | TTFT             | ~9s      | ~2.5s     | **72% lower**   |
| Continuous batching| Wait time        | ~45s     | ~28s      | **44% lower**   |
| Speculative (γ=5)  | TTFT             | ~7.3s    | ~1.3s     | **5.8× faster** |

Results depend on your CPU; the relative improvements are hardware-independent.
