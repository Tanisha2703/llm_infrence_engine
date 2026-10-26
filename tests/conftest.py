"""Shared test setup.

Most tests need the real Qwen2.5-0.5B model in fp32 (about 2 GB). Without this file every test module loaded its own copy and the
suite's memory climbed from ~5 GB to ~11 GB during a run, enough to crash a laptop with little free RAM (measured on a clean
cloud container: results/raw/modal_test_run.txt). Two fixes:

* ``load_model`` for the default model is cached for the whole session, so every module shares one copy. This is safe because
  tests that modify a model (quantization, SmoothQuant) work on ``copy.deepcopy(model)``.
* after each module, run the garbage collector and ask the allocator to return freed memory to the OS.

Other models (the 1.5B speculative target) are loaded fresh and freed after their test.
"""
import ctypes
import functools
import gc

import pytest
import torch

import miniserve.loader as loader

_original_load_model = loader.load_model
_cache: dict = {}


@functools.wraps(_original_load_model)
def _cached_load_model(name_or_path=loader.DEFAULT_MODEL, dtype=torch.float32, device="cpu"):
    if name_or_path != loader.DEFAULT_MODEL:
        return _original_load_model(name_or_path, dtype, device)
    key = (dtype, str(device))
    if key not in _cache:
        _cache[key] = _original_load_model(name_or_path, dtype, device)
    return _cache[key]


loader.load_model = _cached_load_model  # test modules do `from miniserve.loader import load_model` after this file is imported


@pytest.fixture(scope="module", autouse=True)
def _release_memory_between_modules():
    yield
    gc.collect()
    try:  # glibc keeps freed memory in its arenas; hand it back (no-op on Windows/macOS)
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
