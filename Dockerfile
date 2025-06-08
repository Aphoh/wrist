FROM rust:1.87-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy only dependency files for Docker layer caching
COPY pyproject.toml .
COPY uv.lock .

# Create directory structure but don't copy source code
RUN mkdir -p src/sample_model
RUN touch src/sample_model/__init__.py

# Install dependencies
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

RUN /workspace/.venv/bin/python --version

# Source code will be mounted at runtime
CMD ["/bin/sh", "-c", "uv run -m sample_model.simulate"]