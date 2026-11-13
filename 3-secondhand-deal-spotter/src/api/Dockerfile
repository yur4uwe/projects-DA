FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY src/ src/

EXPOSE 8000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
