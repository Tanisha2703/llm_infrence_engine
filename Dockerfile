# miniserve server image (CPU by default).
#
#   docker build -t miniserve .
#   docker run --rm -p 8000:8000 -v miniserve-models:/models miniserve
#
# The model (Qwen2.5-0.5B-Instruct, ~1 GB) is downloaded from the Hugging Face Hub on first start into /models; mount a volume
# there so it is downloaded once. For a GPU image build with a CUDA wheel index instead (not tested here):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t miniserve:gpu .
# and start the container with GPU access plus: --device cuda --dtype bfloat16 --cuda-graphs

FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /app

# 1. heavy dependency first, so source edits do not invalidate this layer
RUN pip install torch --index-url ${TORCH_INDEX}

# 2. the package and its server extras
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[serve]"

# 3. run as an unprivileged user; /models must be writable for the model download
RUN useradd --create-home --uid 10001 miniserve && mkdir -p /models && chown -R miniserve /models
USER miniserve

EXPOSE 8000
# The first start downloads the model, so allow a long warm-up before the check can fail the container.
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["python", "-m", "miniserve.server", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--engine", "paged", "--max-len", "1024"]
