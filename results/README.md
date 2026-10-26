# Results

```
results/
  figures/   the charts and the dashboard screenshot shown in the README
  raw/       the raw data behind every number (JSON, plus a few text logs), listed below
```

Every number quoted in the top-level README and in [docs/RESULTS.md](../docs/RESULTS.md) comes from a file in `raw/`. Each JSON records the script's settings; GPU files also record the GPU, torch and CUDA versions (`_run`). Re-run an experiment with the script named in the first column.

| Files | Script | What |
|---|---|---|
| `figures/*.png` | `make_figures.py` (the Grafana screenshot is a headless-Chrome capture of a live run) | the charts and the dashboard screenshot shown in the top-level README and in RESULTS.md; the charts are regenerated from the JSON files below |
| `kv_cache_cpu_float32.json` | `bench_kv_cache.py` | KV cache vs recomputation, CPU |
| `batching_cpu_float32.json` (`_run1` = an earlier repeat) | `bench_batching.py` | sequential vs static vs continuous batching, CPU |
| `paged_cpu_float32.json` (`_run1`) | `bench_paged.py` | paged KV cache: memory, memory pressure, prefix caching, gather overhead, CPU |
| `speculative_cpu_float32*.json`, `speculative_costs.json` | `bench_speculative.py` | speculative decoding on CPU: main run, repeats, a k=1/3 run, the cost ingredients |
| `quant_ppl.json` | `bench_quant.py --only ppl` | WikiText-2 perplexity and sizes of every quantization scheme |
| `quant_stats.json`, `quant_diag.json`, `quant_gen.json`, `quant_speed.json` | `bench_quant.py`, `diag_outliers.py` | weight/activation statistics, where outliers are, greedy-text agreement, speed |
| `serving_cpu_float32.json`, `serving_cpu_float32_sweep.json` | `bench_serving.py` | scheduler comparison over HTTP, and the open-loop arrival-rate sweep (CPU) |
| `gpu_launch_diag.json` | `diag_gpu_launch.py` | **decode is launch-bound**: step time vs batch size, kernels per step |
| `gpu_batching.json`, `gpu_kv_cache.json`, `gpu_paged.json`, `gpu_speculative.json` | the matching `bench_*.py` | the same experiments on an A10 (bf16) |
| `gpu_parity_fp32.json`, `gpu_parity_bf16.json` | `gpu_parity.py` | do batching, paging and speculation still equal the reference on the GPU? |
| `gpu_serving_{static,continuous,paged}_c16.json`, `..._rep2.json`, `gpu_serving_run1_A10G_console.txt` | `loadtest.py` via `modal_run.py` | HTTP serving, 16 clients; three runs on two GPU variants (run 1 survives only as a console transcript) |
| `gpu_graphs_*.json` | `modal_run.py --suite graphs` | **CUDA-graph decoding vs eager, run in one container on the same GPU**: step time, batching, parity, serving |
| `gpu_profile_*` | `modal_run.py --suite profile` | where the server's own time goes (cProfile of the engine and event-loop threads) |
| `modal_spend.json` | `modal_run.py` | ledger of every cloud run and its estimated cost (list prices) |
| `modal_test_run.txt`, `modal_test_run_before_fix.txt` | `modal_run.py --suite tests` | the test suite on a clean cloud container, with memory after each module, after and before the memory fix |
