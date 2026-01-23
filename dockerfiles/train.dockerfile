# Base image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV UV_PYTHON=python3.13
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
# Copy sourcecode and data
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

# Command to run when container starts
CMD ["uv", "run", "src/mlops_project/train.py"]

