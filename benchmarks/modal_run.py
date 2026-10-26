"""Run miniserve's benchmarks on a rented GPU (Modal), copy the results back, and keep a ledger of what it cost.

  modal run benchmarks/modal_run.py --suite smoke     # ~3 min: GPU info, parity checks, plumbing (cheap)
  modal run benchmarks/modal_run.py --suite core      # kv cache, batching, paging, speculation, serving load tests
  modal run benchmarks/modal_run.py --suite tests     # the whole pytest suite on a CPU container (no GPU), with a memory report
  MINISERVE_GPU=L4 modal run benchmarks/modal_run.py --suite smoke

Credentials: Modal reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment (or ~/.modal.toml). This file loads
``.env`` into the environment but never prints it. Settings (all optional, see .env.example):
  MINISERVE_GPU           GPU type, default A10G
  MINISERVE_BUDGET_USD    refuse to start if the WORST-CASE cost of the suite exceeds this, default 10

What runs where: this script runs on your machine; ``run_jobs`` runs in a Modal container with a GPU. It executes the
same benchmark scripts you run locally (they take --device/--dtype/--out) and returns their JSON, which is saved to
``results/``. The Hugging Face cache lives in a Modal Volume, so models are downloaded once.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Put KEY=VALUE lines of .env into os.environ (values are never printed)."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            v = v.split("#")[0].strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


_load_dotenv()

import modal  # noqa: E402  (after .env so the client sees the token)

GPU = os.environ.get("MINISERVE_GPU", "A10G")
BUDGET = float(os.environ.get("MINISERVE_BUDGET_USD", "10"))
CPU_CORES, MEM_MIB = 4, 16384

# Modal list prices in USD per second, read from modal.com/pricing on 2026-10-21 (check it again before relying on them).
GPU_PRICE = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "A10": 0.000306, "A100": 0.000583,
             "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097}
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222  # per core-second, per GiB-second


def cost_per_second(gpu: str) -> float:
    return GPU_PRICE[gpu] + CPU_CORES * CPU_PRICE + (MEM_MIB / 1024) * MEM_PRICE


app = modal.App("miniserve-bench")
hf_cache = modal.Volume.from_name("miniserve-hf-cache", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(  # the versions this repo was developed and tested with
        "torch==2.14.1", "transformers==5.20.2", "safetensors==0.8.0", "huggingface_hub==2.2.0", "numpy==2.5.4",
        "fastapi==0.156.8", "uvicorn==0.54.0", "prometheus-client", "httpx", "pyarrow",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
)
image = (  # for the GPU benchmarks and GPU tests
    base_image.pip_install("pytest>=8", "pyyaml>=6")
    .add_local_python_source("miniserve")
    .add_local_dir(str(ROOT / "benchmarks"), remote_path="/root/benchmarks")
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/tests")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml")
)
test_image = (  # for the pytest suite: the same libraries plus the tests and the files some tests read
    base_image.pip_install("pytest>=8", "pyyaml>=6")
    .add_local_python_source("miniserve")
    .add_local_dir(str(ROOT / "tests"), remote_path="/root/tests")
    .add_local_dir(str(ROOT / "benchmarks"), remote_path="/root/benchmarks")
    .add_local_dir(str(ROOT / "deploy"), remote_path="/root/deploy")
    .add_local_dir(str(ROOT / "src" / "miniserve"), remote_path="/root/src/miniserve")
    .add_local_file(str(ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml")  # pytest markers live here
    .add_local_file(str(ROOT / "docker-compose.yml"), remote_path="/root/docker-compose.yml")  # read by tests/test_deploy.py
    .add_local_file(str(ROOT / "Dockerfile"), remote_path="/root/Dockerfile")
)
TEST_CPU, TEST_MEM_MIB, TEST_TIMEOUT = 8, 16384, 2400


# ---------------------------------------------------------------------------------------------- what to run
def job(name: str, args: list[str], out: str | None = None, timeout: int = 900) -> dict:
    cmd = ["python", *args] + (["--out", out] if out else [])
    return {"name": name, "cmd": cmd, "out": out, "timeout": timeout}


DEV = ["--device", "cuda", "--dtype", "bfloat16"]
SERVE = "--device cuda --dtype bfloat16 --max-len 1024"
SUITES: dict[str, list[dict]] = {
    "smoke": [
        job("gpu_info", ["-c", "import torch;p=torch.cuda.get_device_properties(0);print(p.name,round(p.total_memory/2**30,1),'GiB',"
                               "'torch',torch.__version__,'cuda',torch.version.cuda)"], timeout=120),
        job("parity_fp32", ["benchmarks/gpu_parity.py", "--device", "cuda", "--dtype", "float32"], "results/raw/gpu_parity_fp32.json", 600),
        job("parity_bf16", ["benchmarks/gpu_parity.py", "--device", "cuda", "--dtype", "bfloat16"], "results/raw/gpu_parity_bf16.json", 600),
        job("micro", ["benchmarks/bench_micro.py", *DEV], None, 600),
    ],
    "core": [
        job("launch_diag", ["benchmarks/diag_gpu_launch.py", *DEV], "results/raw/gpu_launch_diag.json", 600),
        job("kv_cache", ["benchmarks/bench_kv_cache.py", *DEV, "--prompt-lens", "16", "128", "512", "--new-tokens", "64"], "results/raw/gpu_kv_cache.json"),
        job("batching", ["benchmarks/bench_batching.py", *DEV, "--slots", "16", "--requests", "32"], "results/raw/gpu_batching.json", 1500),
        job("paged", ["benchmarks/bench_paged.py", *DEV], "results/raw/gpu_paged.json", 1500),
        job("speculative", ["benchmarks/bench_speculative.py", *DEV, "--ks", "2", "4", "6", "--max-new", "64"], "results/raw/gpu_speculative.json", 1800),
        job("serve_continuous_c16", ["benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine continuous --max-slots 16", "--requests", "64",
                                      "--concurrency", "16", "--out-tokens", "32", "64", "128"], "results/raw/gpu_serving_continuous_c16.json", 900),
        job("serve_paged_c16", ["benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine paged --max-seqs 32", "--requests", "64",
                                 "--concurrency", "16", "--out-tokens", "32", "64", "128", "--shared-prefix-words", "60"], "results/raw/gpu_serving_paged_c16.json", 900),
        job("serve_static_c16", ["benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine static --max-slots 16", "--requests", "64",
                                  "--concurrency", "16", "--out-tokens", "32", "64", "128"], "results/raw/gpu_serving_static_c16.json", 900),
    ],
}

# only the three HTTP serving jobs: cheap, repeatable with --tag to get independent samples (the GPU you are given varies)
SUITES["serving"] = [j for j in SUITES["core"] if j["name"].startswith("serve_")]

# pytest on the GPU: the CUDA-graph tests (skipped everywhere else)
SUITES["gputests"] = [job("cuda_graph_tests", ["-m", "pytest", "tests/test_cuda_graph.py", "-q", "-p", "no:cacheprovider"], None, 900)]

# CUDA graphs vs eager decoding, all in ONE container so both run on the same GPU (the GPU variant you get can differ by ~20%)
# where does the server's own time go? (sampling profiler over the engine and event-loop threads, MINISERVE_PROFILE=1; a short
# timeout on purpose: a hung run must not burn money; the second job prints the report the server wrote when it stopped)
SUITES["profile"] = [
    {"name": "serve_profiled", "out": "results/raw/gpu_profile_serving.json", "timeout": 240,
     "cmd": ["env", "MINISERVE_PROFILE=1", "python", "benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine continuous --max-slots 16 --cuda-graphs",
             "--requests", "64", "--concurrency", "16", "--out-tokens", "32", "64", "128", "--out", "results/raw/gpu_profile_serving.json"]},
    {"name": "thread_profile", "out": "results/raw/gpu_profile_threads.txt", "timeout": 30,
     "cmd": ["cp", "/tmp/profile_threads.txt", "/root/results/raw/gpu_profile_threads.txt"]},
]

SUITES["graphs"] = [
    job("cuda_graph_tests", ["-m", "pytest", "tests/test_cuda_graph.py", "-q", "-p", "no:cacheprovider"], None, 900),
    job("parity_fp32", ["benchmarks/gpu_parity.py", "--device", "cuda", "--dtype", "float32"], "results/raw/gpu_graphs_parity_fp32.json", 600),
    job("parity_bf16", ["benchmarks/gpu_parity.py", "--device", "cuda", "--dtype", "bfloat16"], "results/raw/gpu_graphs_parity_bf16.json", 600),
    job("launch_diag", ["benchmarks/diag_gpu_launch.py", *DEV], "results/raw/gpu_graphs_launch_diag.json", 600),
    job("batching", ["benchmarks/bench_batching.py", *DEV, "--slots", "16", "--requests", "32", "--cuda-graphs"], "results/raw/gpu_graphs_batching.json", 1500),
    job("serve_eager_c16", ["benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine continuous --max-slots 16", "--requests", "64",
                            "--concurrency", "16", "--out-tokens", "32", "64", "128"], "results/raw/gpu_graphs_serving_eager_c16.json", 900),
    job("serve_graphs_c16", ["benchmarks/loadtest.py", "--spawn", f"{SERVE} --engine continuous --max-slots 16 --cuda-graphs", "--requests", "64",
                             "--concurrency", "16", "--out-tokens", "32", "64", "128"], "results/raw/gpu_graphs_serving_graphs_c16.json", 900),
]


# ---------------------------------------------------------------------------------------------- remote side
@app.function(gpu=GPU, image=image, cpu=CPU_CORES, memory=MEM_MIB, volumes={"/cache": hf_cache}, timeout=3 * 3600)
def run_jobs(jobs: list[dict]) -> dict:
    import torch

    t_start = time.time()
    os.makedirs("/root/results/raw", exist_ok=True)
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cc": torch.cuda.get_device_capability(0)}
    done = []
    for j in jobs:
        t0 = time.time()
        try:
            p = subprocess.run(j["cmd"], cwd="/root", capture_output=True, text=True, timeout=j["timeout"])
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT"
        res = None
        if j["out"] and os.path.exists("/root/" + j["out"]):
            res = open("/root/" + j["out"], encoding="utf8").read()
        done.append({"name": j["name"], "returncode": rc, "seconds": time.time() - t0, "stdout": out[-12000:],
                     "stderr": err[-3000:], "result_json": res})
        print(f"[{j['name']}] rc={rc} {time.time() - t0:.0f}s", flush=True)
    hf_cache.commit()  # keep downloaded models for the next run
    return {"info": info, "jobs": done, "seconds": time.time() - t_start}


RSS_PLUGIN = r"""
import sys
import pytest

def _status():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            d[k] = int(v.split()[0]) / 2**20  # kB -> GB
    return d

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    if nextitem is None or nextitem.module is not item.module:  # a module just finished (its fixtures are torn down)
        d = _status()  # some sandboxes do not report the peak (VmHWM), so it is optional
        peak = f", peak {d['VmHWM']:.2f} GB" if "VmHWM" in d else ""
        with open("/tmp/mem.log", "a") as f:  # a file, because pytest captures stdout while tests run
            f.write(f"[mem] after {item.module.__name__}: RSS {d.get('VmRSS', float('nan')):.2f} GB{peak}\n")
"""


@app.function(image=test_image, cpu=TEST_CPU, memory=TEST_MEM_MIB, volumes={"/cache": hf_cache}, timeout=TEST_TIMEOUT + 300)
def run_tests(extra_args: list[str]) -> dict:
    """Run the whole pytest suite on a CPU container and report memory after every test module."""
    import platform

    open("/tmp/rss_plugin.py", "w").write(RSS_PLUGIN)
    env = {**os.environ, "PYTHONPATH": "/root:/tmp"}
    meminfo = open("/proc/meminfo").read().splitlines()[0]
    t0 = time.time()
    try:
        p = subprocess.run(["python", "-m", "pytest", "-q", "-p", "rss_plugin", "-p", "no:cacheprovider", "tests", *extra_args],
                           cwd="/root", env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", "TIMEOUT"
    hf_cache.commit()
    mem_log = open("/tmp/mem.log").read() if os.path.exists("/tmp/mem.log") else ""
    return {"returncode": rc, "stdout": out + "\n" + mem_log, "stderr": err[-3000:], "seconds": time.time() - t0,
            "machine": f"{os.cpu_count()} cpus visible, {meminfo}, {platform.platform()}"}


# ---------------------------------------------------------------------------------------------- local side
def _main_tests() -> None:
    per_s = TEST_CPU * CPU_PRICE + (TEST_MEM_MIB / 1024) * MEM_PRICE
    worst = TEST_TIMEOUT * per_s
    print(f"suite=tests (CPU container, {TEST_CPU} cores, {TEST_MEM_MIB // 1024} GiB)  worst case: ${worst:.2f}   budget: ${BUDGET:.2f}")
    if worst > BUDGET:
        raise SystemExit("worst-case cost exceeds MINISERVE_BUDGET_USD")
    res = run_tests.remote([])
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(exist_ok=True)
    lines = res["stdout"].splitlines()
    tail = "\n".join(lines[-25:])
    mem = [l for l in lines if l.startswith("[mem]")]
    cost = res["seconds"] * per_s
    print(f"\n{res['machine']}\n")
    print("\n".join(mem))
    print("\n" + tail + ("\n--- stderr ---\n" + res["stderr"] if res["returncode"] else ""))
    header = (f"pytest on a Modal CPU container ({res['machine']})\n"
              f"exit code {res['returncode']}, {res['seconds']:.0f}s, estimated ${cost:.3f}\n\n")
    (out_dir / "modal_test_run.txt").write_text(header + "\n".join(mem) + "\n\n" + tail + "\n", encoding="utf8")
    ledger_path = out_dir / "modal_spend.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    ledger.append({"date": time.strftime("%Y-%m-%d %H:%M"), "suite": "tests", "gpu": "none (CPU)",
                   "container_seconds": round(res["seconds"]), "estimated_usd": round(cost, 3),
                   "failed_jobs": [] if res["returncode"] == 0 else ["pytest"]})
    ledger_path.write_text(json.dumps(ledger, indent=2))
    print(f"\ncontainer time {res['seconds']:.0f}s -> estimated ${cost:.3f}. "
          f"Total estimated so far: ${sum(e['estimated_usd'] for e in ledger):.2f}")


@app.local_entrypoint()
def main(suite: str = "smoke", tag: str = ""):
    if suite == "tests":
        return _main_tests()
    if suite not in SUITES:
        raise SystemExit(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
    if GPU not in GPU_PRICE:
        raise SystemExit(f"no price known for GPU {GPU!r}; add it to GPU_PRICE (from modal.com/pricing)")
    jobs = [dict(j) for j in SUITES[suite]]
    if tag:  # results/raw/gpu_x.json -> results/raw/gpu_x_<tag>.json, so a repeat does not overwrite the first run
        for j in jobs:
            if j["out"]:
                new_out = j["out"].replace(".json", f"_{tag}.json")
                j["cmd"] = [new_out if c == j["out"] else c for c in j["cmd"]]
                j["out"] = new_out
    worst = sum(j["timeout"] for j in jobs) * cost_per_second(GPU)
    print(f"suite={suite} gpu={GPU}  worst case (every job hits its timeout): ${worst:.2f}   budget: ${BUDGET:.2f}")
    if worst > BUDGET:
        raise SystemExit("worst-case cost exceeds MINISERVE_BUDGET_USD; raise the budget or run a smaller suite")

    res = run_jobs.remote(jobs)

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(exist_ok=True)
    cost = res["seconds"] * cost_per_second(GPU)
    print(f"\nGPU: {res['info']}\n")
    for r in res["jobs"]:
        flag = "ok " if r["returncode"] == 0 else "FAIL"
        print(f"=== {flag} {r['name']}  ({r['seconds']:.0f}s) ===")
        print(r["stdout"][-2500:] if r["returncode"] == 0 else (r["stdout"][-1500:] + "\n--- stderr ---\n" + r["stderr"]))
        if r["result_json"]:
            target = ROOT / jobs[[j["name"] for j in jobs].index(r["name"])]["out"]
            if target.suffix == ".json":
                data = json.loads(r["result_json"])
                if isinstance(data, dict):
                    data["_run"] = {"gpu": res["info"]["gpu"], "torch": res["info"]["torch"], "cuda": res["info"]["cuda"]}
                target.write_text(json.dumps(data, indent=2))
            else:  # a plain-text artifact (for example the thread profile)
                target.write_text(r["result_json"], encoding="utf8")

    ledger_path = out_dir / "modal_spend.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    ledger.append({"date": time.strftime("%Y-%m-%d %H:%M"), "suite": suite, "gpu": res["info"]["gpu"], "container_seconds": round(res["seconds"]),
                   "estimated_usd": round(cost, 3), "failed_jobs": [r["name"] for r in res["jobs"] if r["returncode"] != 0]})
    ledger_path.write_text(json.dumps(ledger, indent=2))
    print(f"\ncontainer time {res['seconds']:.0f}s -> estimated ${cost:.3f} (list prices; image build time not included). "
          f"Total estimated so far: ${sum(e['estimated_usd'] for e in ledger):.2f}")
