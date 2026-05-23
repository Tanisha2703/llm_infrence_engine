FROM python:3.11-slim

# Build-time deps for llama-cpp-python (C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -e .

COPY src/ ./src/

# Models are mounted at runtime via docker-compose volume
RUN mkdir -p models

ENV MODEL_PATH=models/tinyllama-1.1b-Q4_K_M.gguf
ENV N_CTX=512
ENV N_THREADS=4
ENV MAX_CONCURRENT=2

EXPOSE 8000

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
