# miniserve: an LLM inference engine built from scratch

[![CI](https://github.com/Tanisha2703/llm_infrence_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanisha2703/llm_infrence_engine/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![docker](https://img.shields.io/badge/run-docker%20compose%20up-2496ED)

I implemented the core of an LLM serving stack in PyTorch, **without vLLM or Hugging Face `generate`**, then measured it on a laptop CPU and a rented A10 GPU, found its bottleneck, and fixed it. It runs Qwen2.5-0.5B with my own implementation of the model (checked against Hugging Face) and adds the optimizations production systems use, each verified against the simpler path it replaces:

**KV cache** · **continuous batching** · **paged KV cache with prefix caching and preemption** · **speculative decoding** (exact) · **quantization** (GPTQ, SmoothQuant, int8 KV cache) · **CUDA-graph decoding** · **OpenAI-compatible streaming server** · **Prometheus + Grafana monitoring** · **open/closed-loop load tester** · **Docker** · **GPU benchmarks on Modal**

| At a glance | |
|---|---|
| **3.8x** faster GPU decode step | CUDA graphs remove the launch overhead I measured (24.8 → 6.5 ms at batch 16); output unchanged |
| **~4x** lower p95 time-to-first-token | continuous vs static batching, over HTTP, reproduced on three GPU runs |
| **14 vs 8** concurrent requests | paged KV cache vs fixed slots at equal memory (1.32x throughput on the GPU) |
| **25.4 → 20.8** perplexity | GPTQ vs round-to-nearest int4 at identical size (fp32: 18.1), 7.75x smaller layer weights |
| **159 tests**, 18 injected bugs | every optimization checked token-for-token against the simple path |

![Grafana dashboard of the live server under load](results/figures/grafana_dashboard.png)
<sub>The provisioned Grafana dashboard during a real run: steady load, then an overload burst (the `rejected` line, queue and time-to-first-token spikes).</sub>

## Try it in one command
```bash
docker compose up --build        # or: podman compose up --build
```
This starts the **server** (http://localhost:8000), **Prometheus** (:9090) and **Grafana** (http://localhost:3000, login `admin`/`admin`, dashboard "miniserve"). The first start downloads the model (~1 GB) into a volume; the API answers once `/health` is green (about a minute). Then:
```bash
curl localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"prompt": "The three primary colors are", "max_tokens": 24, "stream": true}'
```
The fp32 CPU model needs about 4 GB of free memory for the container (`MINISERVE_DTYPE=bfloat16 docker compose up` uses half). Without Docker: see [Run it without Docker](#run-it-without-docker).

## What the measurements show

**1. The bottleneck: decode on a GPU is launch-bound, not memory-bound.** One decode step of the eager engine launches 1,358 kernels and takes 18-25 ms *whether the batch is 1 or 32*; 97-98% of that is the CPU issuing launches, against a memory-bandwidth ideal of 2 ms.

**2. The fix: CUDA graphs.** I record the whole decode step once per batch size and replay it with a single launch. On the same A10, in the same container, the decode step drops from **24.8 ms to 6.5 ms at batch 16 (3.8x)**, engine throughput on a request burst rises **416 → 774 tok/s (1.86x)**, and the output is unchanged (exact in fp32 on the GPU; 8 GPU tests).

![eager vs CUDA graph decode step and throughput](results/figures/gpu_cuda_graphs.png)

Over HTTP the gain is smaller (443 → 598 tok/s, 1.35x, in the same-container comparison; a later run measured 715). I had guessed the server's Python code was the culprit, so I profiled the live server with a sampling profiler: the **event loop is 95% idle**; the engine thread spends **66% of its time waiting for the GPU** and **17% in prefill, which is still eager and launch-bound**. The server is not the bottleneck; batching prefills is the next optimization.

**3. Continuous batching** (served over HTTP, 16 clients, A10 / A10G, eager decode): **1.22-1.25x throughput and 3.8-4.0x lower p95 time-to-first-token** than static batching, reproduced across three GPU runs. On the CPU it gave 2.0-2.1x over sequential serving; the paged KV cache held **14 concurrent requests instead of 8 at equal memory** (1.32x throughput on the GPU).

![continuous vs static batching over HTTP on a GPU](results/figures/gpu_serving.png)

**4. GPTQ** recovers most of the accuracy lost to 4-bit rounding at identical size (perplexity 25.4 → 20.8 at group size 128; fp32 baseline 18.1) with layer weights 7.75x smaller. A diagnostic found why 8-bit activation quantization fails on this model: first-token "massive activations" (values of 1,000-1,600 vs typical 2-12) at the `down_proj` inputs; per-token scales fix it (79.5 → 18.5).

![RTN vs GPTQ perplexity](results/figures/quantization_quality.png)

**5. Negative results are reported, not hidden.** Quantization gives **no speedup** without fused kernels (dequantizing on every call is 4-8x *slower*); speculative decoding with a draft model *loses* on the eager GPU engine (the draft step costs 82% of a target step, so 0.62-0.85x), while a free n-gram drafter wins on copy-heavy text (1.44-1.56x). Both follow from finding 1.

Full tables, methodology and caveats: [docs/RESULTS.md](docs/RESULTS.md). Design and decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Raw data for every number: [results/](results/README.md).

## Engineering rigor
* **159 tests**, passing on a laptop and on a clean Linux container. Every optimization is tested for token-for-token equality with the simpler path (HF model = this model = cached = batched = paged = speculative = CUDA-graph = served over HTTP, fp32); quantization is tested against measured error bounds and perplexity.
* **Bug injection:** I deliberately broke the risky code (18 injected bugs across quantization, the server, the load tester, monitoring config and batched sampling) to confirm tests fail. It found weak tests, which I strengthened.
* **Correctness re-checked on the GPU:** exact in fp32 for every scheduling path including CUDA graphs; the bf16 rounding drift is reported, not hidden.
* **The monitoring is verified, not assumed:** all 26 dashboard queries were executed against real Prometheus and Grafana containers, an alert was made to fire, and a test fails if a dashboard query references a metric the server does not export.
* **Tested somewhere other than my laptop:** running the suite on a clean cloud machine exposed a CPU-dependent assertion and a memory problem (12.3 GB peak); both fixed (5.6 GB, 1.6x faster). Total cloud spend for every GPU and CPU run in this project: about $1.1 (estimated at list prices; ledger in `results/raw/modal_spend.json`).

## Run it without Docker
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,eval,serve]"
pytest -q                                                    # all tests (-m "not heavy" skips the one that needs ~9 GB RAM)

python -m miniserve.server --engine paged                    # http://127.0.0.1:8000  (add --cuda-graphs on a GPU with --device cuda)
python benchmarks/loadtest.py --spawn "--engine paged" --rate 0.5 --requests 30   # open-loop load test; starts its own server
```
Shortcuts for these commands are in the [Makefile](Makefile) (`make test`, `make serve`, `make up`, ...). **GPU runs:** copy `.env.example` to `.env`, add a Modal token, then `modal run benchmarks/modal_run.py --suite smoke` (a worst-case cost estimate is checked against `MINISERVE_BUDGET_USD` before anything runs; other suites: `core`, `graphs`, `tests`, `gputests`).

## Limits (read before quoting any number)
* **No custom CUDA/Triton kernels:** paged attention is a gather and quantization is simulated in floating point, so no quantization speed claim is made. CUDA graphs cover the slot-cache continuous engine only (not the paged engine, prefill or speculative decoding).
* After graphs the decode step is still ~3x above the memory-bandwidth ideal, and prefill is still eager (17% of engine-thread time under load). I make no comparison with tuned systems like vLLM.
* One small model (0.5B), short contexts (≤ ~600 tokens), WikiText-2 perplexity only. GPU numbers come from one GPU family (two variants differing by ~20%), 64 requests per run, so p99 ≈ maximum.
* The GitHub Actions workflow has not run yet. The Docker setup was tested with Podman (bf16 inside a 2 GiB VM), not with Docker itself, and the GPU image build argument is untested.

## Repository
| Path | |
|---|---|
| `src/miniserve/` | model, loader, caches, engines, CUDA-graph decoder, speculative decoding, quantization, server |
| `tests/` | 159 tests |
| `benchmarks/` | one script per experiment, load tester, GPU runner (`modal_run.py`), figure generator |
| `results/` | everything measured: raw JSON of every reported number, the Modal cost ledger, and `figures/` (the charts and dashboard screenshot used above); [index](results/README.md) |
| `deploy/` | Prometheus config + alert rules, Grafana provisioning + dashboard |
| `Dockerfile`, `docker-compose.yml` | the container image and the server + Prometheus + Grafana stack |
| `docs/` | [RESULTS.md](docs/RESULTS.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), [papers.md](docs/papers.md) (which papers are implemented and what is not) |

## References
Orca (iteration-level scheduling), vLLM / PagedAttention, Leviathan et al. and Chen et al. (speculative sampling), GPTQ, SmoothQuant, LLM.int8(), FlashAttention (via PyTorch SDPA), CUDA Graphs. Details in [docs/papers.md](docs/papers.md).
