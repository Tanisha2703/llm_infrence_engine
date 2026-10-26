# Architecture and design decisions

## Components

```
 client ── HTTP / SSE ──▶  server.py            FastAPI app: validation, queue limit (429), streaming, cancel-on-disconnect,
                              │                 /health, /metrics (Prometheus)
                              │ mailbox (engine → loop)      add_request / cancel (loop → engine)
                              ▼
                        engine thread           Engine (fixed slots)  |  PagedEngine (block pool, prefix cache, preemption)
                              │                 one scheduler step = admit + prefill new requests, one batched decode
                              ▼
                        model(ids, cache=view)  model.py: Qwen2 from scratch (GQA attention, RoPE, SwiGLU, RMSNorm, tied head)
                              │
                 ┌────────────┼──────────────┐
            cache.py       paged.py        quant.py
        KVCache, SlotCache  PagedCache   QuantLinear, GPTQ, SmoothQuant, Int8KVCache
```

| Module | Responsibility |
|---|---|
| `model.py`, `loader.py` | The transformer, written from scratch and loaded from Hugging Face safetensors; verified against `transformers` (logits and greedy text). |
| `cache.py`, `paged.py` | KV caches behind **one interface** (`prepare / positions / update / mask / advance`), so the model code is identical for a plain cache, a slot cache and a paged cache. |
| `generate.py`, `sampler.py` | Prefill + decode loop as a generator (what streaming needs); greedy, temperature, top-k, top-p, seeded. |
| `engine.py`, `paged_engine.py` | Request lifecycle and the per-step scheduler: static batching (baseline) and continuous batching; the paged engine adds block-level admission, prefix caching and recompute preemption. |
| `speculative.py` | Exact speculative sampling (acceptance rule plus residual resampling), draft-model and n-gram drafters, KV rollback. |
| `quant.py` | Round-to-nearest quantizers (int8/int4/int3; per-tensor, per-channel, group-wise; symmetric and asymmetric; packed int4), GPTQ, SmoothQuant, activation quantization, an int8 KV cache with static per-channel key scales. |
| `graph_decode.py` | CUDA-graph decoding: one graph per batch size over a static-shape cache view (`StaticSlotBatch` in `cache.py`); padding rows go to a scratch slot. |
| `server.py`, `metrics.py`, `profiling.py` | HTTP serving, metrics, and an opt-in sampling profiler (`MINISERVE_PROFILE=1`). |
| `deploy/`, `Dockerfile`, `docker-compose.yml` | Prometheus scrape config and alert rules, Grafana provisioning and dashboard; the image and the one-command stack (server + Prometheus + Grafana). |

## One request, end to end
1. **Event loop:** FastAPI validates the JSON (pydantic bounds); `submit()` checks engine health and the queue limit (`429 + Retry-After` if full), builds a `Request` with a private mailbox, hands it to the engine, wakes the engine thread.
2. **Engine thread:** the next `step()` admits it (paged engine: only if enough free KV blocks, reusing any cached prefix blocks), prefills the prompt, then decodes one token per step together with every other running request (one batched forward pass).
3. **Each token** is pushed to the request's mailbox with `loop.call_soon_threadsafe`; the handler's async generator turns token ids into text (an incremental detokenizer withholds half-finished UTF-8 characters) and writes a server-sent event.
4. **On finish** (stop token, length, client gone, error) a callback records latency histograms and posts the final chunk with token usage.

## Key design decisions

| Decision | Why | Evidence |
|---|---|---|
| The model runs on **its own thread**, not on the web framework's event loop | A forward pass blocks for tens of milliseconds; on the loop it would stall health checks, new connections and token streaming. | Demo: a health request arriving mid-step is served ~216 ms late on a blocked loop vs ~10 ms with a separate thread. |
| **Measure, then fix, then re-measure** (CUDA graphs) | The launch-bound finding (1,358 launches, 97-98% CPU time) pointed at the fix; the fix is verified equal to the eager path and measured in the same container. | Decode step 3.1-4.2x faster; engine 1.86x; the profiler then showed the server is *not* the next bottleneck (event loop 95% idle), prefill is. |
| **One cache interface** for all cache types | Lets batching, paging and quantized caches be swapped in without touching attention code, and be tested against each other. | Same model code runs plain, slot, paged and int8 caches. |
| **Every optimization is tested against the simpler path** | A fast wrong implementation is worthless. Chain: Hugging Face = this model = cached = batched = paged = speculative = served over HTTP (fp32). Quantization, which changes outputs, is tested against measured error bounds and perplexity. | 148 tests; GPU re-check in fp32 (exact) and bf16 (rounding drift reported). |
| **Bug injection** on risky logic | A passing suite proves nothing if the tests cannot fail. | 6 quantization bugs and 11 server/load-tester/monitoring-config bugs injected; the one miss (a TTFT measured at the wrong event) was fixed by strengthening its test. |
| **Admission by free KV blocks** (paged engine) with preemption by recompute | Reserving worst-case memory per request wastes ~50% of the pool; blocks let more requests run, and recompute preemption is simpler than swapping. | 14 vs 8 concurrent requests at equal memory. |
| **Backpressure** (`429`) instead of an unbounded queue | An unbounded queue makes every request slow under overload; a bounded one sheds the excess and keeps accepted requests fast. | Open-loop sweep on CPU: TTFT p95 stays bounded while `429`s appear. |
| Gauges read engine state **at scrape time** | A hand-updated gauge can drift from reality; a callback cannot. | `miniserve_requests_waiting`, KV blocks, `engine_up`. |
| Dashboard and alert queries are **tested against the exported metric names** | A renamed metric would otherwise show up only as an empty panel. | `tests/test_deploy.py`. |
| Honest negative results are kept | They are findings, and they explain the other numbers. | No quantization speedup without kernels; draft-model speculation loses on an eager GPU engine; launch-bound decode. |

## Threading and failure model
Two kinds of thread (event loop, one engine thread) communicate only through the engine's lock-protected waiting queue (loop → engine) and per-request `asyncio.Queue` mailboxes fed via `call_soon_threadsafe` (engine → loop). Cancellation is a flag set from the loop and acted on by the engine thread at the next step boundary, so the loop never touches engine state. If `step()` raises, every waiting and running request is failed (`500`), `/health` returns `503`, `miniserve_engine_up` drops to 0 (an alert fires) and new requests get `503`: requests never hang.

## What is deliberately not here
Custom CUDA/Triton kernels (paged attention is a gather, quantization is simulated in floating point), CUDA graphs, tensor/pipeline parallelism, chunked prefill, authentication and TLS, Alertmanager. See `docs/RESULTS.md` for what the measurements say about these gaps.
