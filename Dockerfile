# syntax=docker/dockerfile:1.4
# Stage 1: Build & dependency resolution with uv
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY pyproject.toml uv.lock* ./

# Install dependencies into virtualenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip sync --python /app/.venv pyproject.toml || uv pip install -r pyproject.toml --python /app/.venv

# Stage 2: Minimal runtime image
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Install runtime system packages (e.g. for pdf/ocr processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYARROW_IGNORE_TIMEZONE="1" \
    PYTHONUNBUFFERED=1

# Copy source code, fallback data, and configurations
COPY src/ /app/src/
COPY data/ /app/data/
COPY pyproject.toml /app/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
