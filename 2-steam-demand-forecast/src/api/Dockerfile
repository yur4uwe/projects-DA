FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies required for Prophet / Stan / LightGBM
# Delete apt lists to keep the image lean
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    curl \ 
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY models/ models/
# Copy only processed parquet features
COPY data/ data/

EXPOSE 8000 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
