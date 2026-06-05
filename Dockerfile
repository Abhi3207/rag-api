# ---- Base stage ----
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps needed by chromadb (SQLite, build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from base
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# ChromaDB data lives in a volume so it survives container restarts
VOLUME ["/app/chroma_db"]

EXPOSE 8000

# Run with uvicorn; use 0.0.0.0 so Docker exposes the port correctly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
