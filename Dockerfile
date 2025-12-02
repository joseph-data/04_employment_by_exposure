FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps required by pandas/numpy/plotly stack
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 shiny
USER shiny

COPY --chown=shiny:shiny . .

EXPOSE 7860

CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "7860", "app.py"]
