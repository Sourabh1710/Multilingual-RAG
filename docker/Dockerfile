# STAGE 1: The Builder (Install & Compile)
FROM python:3.12-slim AS builder

# Copy the pre-compiled, standalone uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:$PATH"

WORKDIR /app

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1

# Copy the project definition files
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies cleanly (cached by Docker layers)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the source code
COPY src/ ./src

# Install the project wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# STAGE 2: The Runtime (Small & Secure)
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy only the compiled virtual environment from the Builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the source code into the runtime container
COPY src/ ./src

# Ensure the environment uses the virtual environment packages
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI's default port
EXPOSE 8000

# Run Uvicorn asynchronously to serve the API
CMD ["uvicorn", "src.rag_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
