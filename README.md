# 🧠 RAG API

A production-grade **Retrieval-Augmented Generation** API built with **FastAPI**, **ChromaDB**, and **Ollama**. Store personal profiles as vector embeddings and query them with natural language — powered by a local LLM.

## Architecture

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│   Client     │──────▶│   FastAPI     │──────▶│   Ollama    │
│  (REST)      │◀──────│   main.py     │◀──────│   (LLM)     │
└─────────────┘       └──────┬───────┘       └─────────────┘
                             │
                      ┌──────▼───────┐
                      │  ChromaDB    │
                      │ (Vector DB)  │
                      └──────────────┘
```

## Features

- **Document Management** — Add, list, and delete profile chunks via JSON or file upload
- **Single-turn RAG** (`POST /ask`) — Ask a question, get a context-aware answer
- **Multi-turn Chat** (`POST /chat`) — Conversational RAG with message history
- **Health Check** (`GET /health`) — Monitor Ollama & ChromaDB connectivity
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
# Edit .env to customise Ollama URL, models, etc.
```

### 3. Build the knowledge base

```bash
python build_knowledge_base.py                         # uses profile.txt
python build_knowledge_base.py -f resume.txt -u alice  # custom file & user
```

### 4. Run the API

```bash
uvicorn main:app --reload
```

The interactive docs are at **http://localhost:8000/docs**.

### 5. Docker (optional)

```bash
docker build -t rag-api .
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 rag-api
```

## API Reference

### System

| Method | Endpoint   | Description                      |
|--------|-----------|----------------------------------|
| GET    | `/health` | Ollama & ChromaDB health status  |

### Documents

| Method | Endpoint                  | Description                          |
|--------|--------------------------|--------------------------------------|
| POST   | `/documents`             | Add profile text as chunks (JSON)    |
| POST   | `/documents/upload`      | Upload a `.txt` file                 |
| GET    | `/documents`             | List stored chunks                   |
| DELETE | `/documents/{user_name}` | Delete all chunks for a user         |

### RAG

| Method | Endpoint | Description                        |
|--------|----------|------------------------------------|
| POST   | `/ask`   | Single-turn RAG query              |
| POST   | `/chat`  | Multi-turn conversational RAG      |

### Example — Ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Yash'\''s skills?", "user": "default"}'
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

## Environment Variables

| Variable            | Default              | Description                     |
|---------------------|----------------------|---------------------------------|
| `OLLAMA_URL`        | `http://localhost:11434` | Ollama server address        |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:0.5b`      | LLM model for generation       |
| `OLLAMA_EMBED_MODEL`| `nomic-embed-text`   | Model for embeddings            |
| `CHROMA_DB_PATH`    | `./chroma_db`        | ChromaDB persistence directory  |
| `COLLECTION_NAME`   | `personal_profile`   | ChromaDB collection name        |
| `DEFAULT_N_RESULTS` | `3`                  | Default retrieval count         |
| `CORS_ORIGINS`      | `["*"]`              | Allowed CORS origins            |

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

## Project Structure

```
rag-api/
├── main.py                  # FastAPI application (all endpoints)
├── config.py                # Centralised settings (env / .env)
├── build_knowledge_base.py  # CLI tool to ingest text files
├── profile.txt              # Sample profile data
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build
├── .env.example             # Environment variable template
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Pytest suite
└── README.md
```

## License

MIT
