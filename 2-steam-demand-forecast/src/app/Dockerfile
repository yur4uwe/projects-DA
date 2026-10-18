FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ src/

EXPOSE 8501

HEALTHCHECK CMD python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health")' || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
