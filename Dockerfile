FROM python:3.13-slim AS base

# Keep Python output unbuffered (better logs), avoid creating .pyc files,
# and make pip installs more deterministic in containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (some Python packages may compile native extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run as a non-root user (required/recommended by many platforms incl. HF Spaces).
RUN useradd -m -u 1000 shiny
USER shiny

# Copy the application code and assets.
COPY --chown=shiny:shiny . .

# Hugging Face Spaces default container port.
EXPOSE 7860

# Start the Shiny for Python app.
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "7860", "app.py"]
