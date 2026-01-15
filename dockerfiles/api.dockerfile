FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV UV_PYTHON=python3.13
ENV PYTHONPATH=/src
WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/

COPY mlops-api-sa.json /mlops-api-sa.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/mlops-api-sa.json

RUN uv sync --locked --no-cache --no-install-project

CMD ["uv", "run", "--no-sync", "uvicorn", "mlops_project.api:app", "--host", "0.0.0.0", "--port", "8000"]

