# 🧠 RAG API

A production-grade **Retrieval-Augmented Generation** API built with **FastAPI**, **ChromaDB**, and **Ollama**. Store personal profiles as vector embeddings and query them with natural language — powered by a local LLM.

## Architecture

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│   Client     │──────▶│   FastAPI     │──────▶│   Ollama    │
│  (REST/SSE)  │◀──────│   (async)     │◀──────│   (LLM)     │
└─────────────┘       └──────┬───────┘       └─────────────┘
                             │
                      ┌──────▼───────┐
                      │  ChromaDB    │
                      │ (Vector DB)  │
                      └──────────────┘
```

## Features

- **Document Management** — Add, list, and delete profile chunks via JSON or file upload
- **Pluggable Chunking** — Paragraph, recursive, or semantic chunking strategies
- **Single-turn RAG** (`POST /ask`) — Ask a question, get a context-aware answer
- **Multi-turn Chat** (`POST /chat`) — Conversational RAG with message history
- **Streaming Responses** (`POST /ask/stream`, `/chat/stream`) — Real-time SSE token streaming
- **Health Check** (`GET /health`) — Monitor Ollama & ChromaDB connectivity
- **Admin Analytics** (`GET /admin/stats`) — Knowledge base statistics and user metrics
- **API Key Auth** — Optional `X-API-Key` authentication (opt-in)
- **Rate Limiting** — Configurable per-endpoint rate limits via SlowAPI
- **Fully Async** — Non-blocking Ollama calls, ChromaDB queries, and health checks
- **User Filtering** — Scope queries to a specific user's documents
- **CORS Enabled** — Ready for frontend integration
- **Configurable** — All settings via environment variables or `.env` file

## Prerequisites

| Dependency | Version |
|------------|---------|
| Python     | 3.10+   |
| Ollama     | Latest  |

**Ollama models required:**

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Abhi3207/rag-api.git
cd rag-api
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env to customise Ollama URL, models, auth, rate limits, etc.
```

### 3. Build the knowledge base

```bash
python build_knowledge_base.py                                     # uses profile.txt
python build_knowledge_base.py -f resume.txt -u alice              # custom file & user
python build_knowledge_base.py --strategy recursive --chunk-size 300  # recursive chunking
```

### 4. Run the API

```bash
uvicorn src.rag_api.app:app --reload
```

The interactive docs are at **http://localhost:8000/docs**.

### 5. Docker (optional)

```bash
docker build -t rag-api .
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 rag-api
```

## API Reference

### System

| Method | Endpoint        | Description                      |
|--------|----------------|----------------------------------|
| GET    | `/health`      | Ollama & ChromaDB health status  |
| GET    | `/admin/stats` | Knowledge base statistics        |

### Documents

| Method | Endpoint                  | Description                                  |
|--------|--------------------------|----------------------------------------------|
| POST   | `/documents`             | Add profile text as chunks (JSON body)       |
| POST   | `/documents/upload`      | Upload a `.txt` file                         |
| GET    | `/documents`             | List stored chunks                           |
| DELETE | `/documents/{user_name}` | Delete all chunks for a user                 |

### RAG

| Method | Endpoint       | Description                        |
|--------|---------------|------------------------------------|
| POST   | `/ask`        | Single-turn RAG query              |
| POST   | `/ask/stream` | Single-turn RAG with SSE streaming |
| POST   | `/chat`       | Multi-turn conversational RAG      |
| POST   | `/chat/stream`| Multi-turn RAG with SSE streaming  |

### Chunking Strategies

When adding documents, specify the `chunk_strategy` parameter:

| Strategy    | Description                                                      |
|-------------|------------------------------------------------------------------|
| `paragraph` | Split on blank lines (`\n\n`). Default, backward-compatible.     |
| `recursive` | Recursive character splitting with configurable size & overlap.  |
| `semantic`  | Sentence-aware grouping that keeps related sentences together.   |

### Example — Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Yash'\''s skills?", "user": "default"}'
```

### Example — Streaming

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about Yash.", "user": "default"}'
```

### Example — Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about Yash."},
      {"role": "assistant", "content": "Yash is interested in cloud computing and AI."},
      {"role": "user", "content": "What technologies does he know?"}
    ],
    "user": "default"
  }'
```

### Example — Recursive Chunking

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "alice",
    "content": "Long document text...",
    "chunk_strategy": "recursive",
    "chunk_size": 300,
    "chunk_overlap": 50
  }'
```

### Example — API Key Auth

```bash
# When API_KEY_ENABLED=true in .env
curl -X GET http://localhost:8000/documents \
  -H "X-API-Key: your-secret-key"
```

## Environment Variables

| Variable               | Default                  | Description                          |
|------------------------|--------------------------|--------------------------------------|
| `OLLAMA_URL`           | `http://localhost:11434` | Ollama server address                |
| `OLLAMA_CHAT_MODEL`    | `qwen2.5:0.5b`          | LLM model for generation            |
| `OLLAMA_EMBED_MODEL`   | `nomic-embed-text`       | Model for embeddings                 |
| `CHROMA_DB_PATH`       | `./chroma_db`            | ChromaDB persistence directory       |
| `COLLECTION_NAME`      | `personal_profile`       | ChromaDB collection name             |
| `DEFAULT_N_RESULTS`    | `3`                      | Default retrieval count              |
| `DEFAULT_CHUNK_STRATEGY`| `paragraph`             | Default chunking strategy            |
| `DEFAULT_CHUNK_SIZE`   | `500`                    | Default chunk size (chars)           |
| `DEFAULT_CHUNK_OVERLAP`| `50`                     | Default overlap between chunks       |
| `RATE_LIMIT_RAG`       | `60/minute`              | Rate limit for RAG endpoints         |
| `RATE_LIMIT_DOCS`      | `200/minute`             | Rate limit for document endpoints    |
| `API_KEY_ENABLED`      | `false`                  | Enable API key authentication        |
| `API_KEY`              | (empty)                  | The API key value (when enabled)     |
| `CORS_ORIGINS`         | `["*"]`                  | Allowed CORS origins                 |
| `LOG_LEVEL`            | `INFO`                   | Logging level                        |

## SSE Streaming Protocol

The streaming endpoints (`/ask/stream`, `/chat/stream`) use Server-Sent Events with these event types:

| Event     | Data                              | Description                   |
|-----------|-----------------------------------|-------------------------------|
| `context` | JSON array of context documents   | Retrieved documents used      |
| `token`   | Raw token string                  | Individual LLM token          |
| `done`    | Empty string                      | Signals stream completion     |

## Testing

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src/rag_api --cov-report=term-missing
```

## Project Structure

```
rag-api/
├── src/
│   └── rag_api/
│       ├── __init__.py              # Package init, version
│       ├── app.py                   # FastAPI app factory, lifespan, middleware
│       ├── config.py                # Centralised settings (env / .env)
│       ├── database.py              # ChromaDB client & collection (lazy init)
│       ├── models.py                # Pydantic request/response models
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── health.py            # GET /health, GET /admin/stats
│       │   ├── documents.py         # Documents CRUD
│       │   └── rag.py               # /ask, /ask/stream, /chat, /chat/stream
│       ├── services/
│       │   ├── __init__.py
│       │   ├── chunking.py          # Pluggable chunking strategies
│       │   ├── retrieval.py         # Context retrieval from ChromaDB
│       │   └── llm.py               # Async Ollama interaction + streaming
│       └── middleware/
│           ├── __init__.py
│           ├── auth.py              # Optional API key authentication
│           └── rate_limit.py        # SlowAPI rate limiting
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── test_health.py               # Health & admin stats tests
│   ├── test_documents.py            # Document CRUD tests
│   ├── test_rag.py                  # Ask & chat tests
│   ├── test_streaming.py            # SSE streaming tests
│   ├── test_chunking.py             # Chunking strategy unit tests
│   └── test_auth.py                 # Auth middleware tests
├── build_knowledge_base.py          # CLI tool to ingest text files
├── profile.txt                      # Sample profile data
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build (multi-stage + HEALTHCHECK)
├── .env.example                     # Environment variable template
├── .gitignore
└── README.md
```

## Upgrading from v1

If you're upgrading from v1, note these breaking changes:

1. **Start command changed**: `uvicorn main:app` → `uvicorn src.rag_api.app:app`
2. **Project restructured**: `main.py` has been split into `src/rag_api/` package
3. **New dependencies**: `sse-starlette` and `slowapi` — run `pip install -r requirements.txt`
4. **Docker CMD updated**: Dockerfile now points to the new module path

The old `main.py` and `config.py` at the project root are no longer used by the new structure.

## License

MIT
