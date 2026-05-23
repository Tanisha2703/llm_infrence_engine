"""FastAPI serving endpoint with KV cache, scheduling, and Prometheus metrics."""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

from src.server.inference import InferenceEngine
from src.optimizations.kv_cache import LRUCache
from src.optimizations.batching import ContinuousBatchScheduler

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("llm_requests_total", "Total inference requests")
CACHE_HIT = Counter("llm_cache_hits_total", "KV cache hits")
CACHE_MISS = Counter("llm_cache_misses_total", "KV cache misses")
TTFT_HIST = Histogram(
    "llm_ttft_seconds",
    "Time to first token",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
TPS_GAUGE = Gauge("llm_tokens_per_second", "Current tokens per second")
CACHE_HIT_RATE = Gauge("llm_cache_hit_rate", "Rolling cache hit rate")

# ---------------------------------------------------------------------------
# Global engine / scheduler (initialised in lifespan)
# ---------------------------------------------------------------------------
_engine: Optional[InferenceEngine] = None
_cache: Optional[LRUCache] = None
_scheduler: Optional[ContinuousBatchScheduler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _cache, _scheduler
    model_path = os.environ.get(
        "MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    n_ctx = int(os.environ.get("N_CTX", "512"))
    n_threads = int(os.environ.get("N_THREADS", "4"))

    _engine = InferenceEngine(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
    _cache = LRUCache(capacity=256)
    _scheduler = ContinuousBatchScheduler(
        inference_fn=_engine.generate,
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", "2")),
    )
    yield


app = FastAPI(title="LLM Inference Optimizer", lifespan=lifespan)

# Mount Prometheus /metrics endpoint
app.mount("/metrics", make_asgi_app())


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    use_cache: bool = True


class GenerateResponse(BaseModel):
    text: str
    ttft: float
    total_time: float
    token_count: int
    tps: float
    cache_hit: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    REQUEST_COUNT.inc()

    # KV cache lookup
    cache_hit = False
    if req.use_cache and _cache:
        cached = _cache.get(req.prompt)
        if cached is not None:
            CACHE_HIT.inc()
            CACHE_HIT_RATE.set(_cache.hit_rate)
            cache_hit = True
            return GenerateResponse(
                text=cached,
                ttft=0.0,
                total_time=0.0,
                token_count=0,
                tps=0.0,
                cache_hit=True,
            )
        CACHE_MISS.inc()

    # Schedule inference
    from src.optimizations.batching import Request as InferRequest

    ir = InferRequest(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    result = _scheduler.submit(ir)

    # Update cache
    if req.use_cache and _cache and result:
        _cache.put(req.prompt, result["text"])

    # Record metrics
    TTFT_HIST.observe(result["ttft"])
    TPS_GAUGE.set(result["tps"])
    CACHE_HIT_RATE.set(_cache.hit_rate if _cache else 0.0)

    return GenerateResponse(
        text=result["text"],
        ttft=result["ttft"],
        total_time=result["total_time"],
        token_count=result["token_count"],
        tps=result["tps"],
        cache_hit=cache_hit,
    )
