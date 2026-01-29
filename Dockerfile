# ------------------------------- Builder Stage ------------------------------ #
FROM python:3.12-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh -o /install.sh \
  && chmod 755 /install.sh \
  && /install.sh \
  && rm -f /install.sh

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install deps from lockfile (cache uv downloads for faster rebuilds)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


## ------------------------------ Production Stage ---------------------------- ##
FROM python:3.12-slim-bookworm AS production

WORKDIR /app

# Environment set-up
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only what the app needs at runtime
COPY app.py ./app.py
COPY src ./src
#COPY www ./www
COPY css ./css

# If the app needs data at runtime.
COPY data ./data

# Requirement for deployment at hf
EXPOSE 7860
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
