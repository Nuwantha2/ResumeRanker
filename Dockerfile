# Dockerfile — pinned to Python 3.11 and minimal build deps
FROM python:3.11-slim

# Install system packages needed for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc libpq-dev curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first for caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip & install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python deps
RUN pip install -r /app/requirements.txt

# Copy application code
COPY . /app

ENV PORT=8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
