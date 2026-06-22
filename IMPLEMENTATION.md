# 🧠 RAG API — Detailed Implementation Guide

> **Purpose**: This document covers every module, function, design decision, and data flow in the RAG API project. It is written to be interview-ready — each section explains the *what*, *why*, and *how* at a depth suitable for technical discussions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack — Deep Dive](#2-tech-stack--deep-dive)
3. [Project Structure & Module Map](#3-project-structure--module-map)
4. [Configuration System (`config.py`)](#4-configuration-system)
5. [Application Factory (`app.py`)](#5-application-factory)
6. [Database Layer (`database.py`)](#6-database-layer)
7. [Pydantic Models (`models.py`)](#7-pydantic-models)
8. [Chunking Service (`services/chunking.py`)](#8-chunking-service)
9. [Retrieval Service (`services/retrieval.py`)](#9-retrieval-service)
10. [LLM Service (`services/llm.py`)](#10-llm-service)
11. [Document Routes (`routes/documents.py`)](#11-document-routes)
12. [RAG Routes (`routes/rag.py`)](#12-rag-routes)
13. [Health & Admin Routes (`routes/health.py`)](#13-health--admin-routes)
14. [Authentication Middleware (`middleware/auth.py`)](#14-authentication-middleware)
15. [Rate Limiting Middleware (`middleware/rate_limit.py`)](#15-rate-limiting-middleware)
16. [CLI Ingestion Tool (`build_knowledge_base.py`)](#16-cli-ingestion-tool)
17. [Docker & Deployment](#17-docker--deployment)
18. [Testing Strategy](#18-testing-strategy)
19. [SSE Streaming Protocol](#19-sse-streaming-protocol)
20. [End-to-End Data Flow](#20-end-to-end-data-flow)
21. [Interview Q&A Bank](#21-interview-qa-bank)

---

## 1. Project Overview

The RAG API is a **Retrieval-Augmented Generation** system that:

1. **Stores** personal profile text as vector embeddings in ChromaDB.
2. **Retrieves** the most semantically relevant chunks for a user's question using cosine similarity search.
3. **Augments** an LLM prompt with the retrieved context.
4. **Generates** a natural language answer using a locally hosted Ollama LLM.

### Why RAG?

Large Language Models (LLMs) have a knowledge cutoff and cannot access private/custom data. RAG solves this by:
- **Retrieval**: Fetching relevant documents from a vector database at query time.
- **Augmentation**: Injecting those documents into the LLM's prompt as context.
- **Generation**: The LLM generates an answer *grounded in the retrieved context*, reducing hallucinations.

### Key Design Principles

| Principle | Implementation |
|---|---|
| **Fully Async** | All I/O (Ollama calls, ChromaDB queries) runs asynchronously via `asyncio.to_thread` or native async clients |
| **Modular Architecture** | Clean separation: routes → services → database, each layer independently testable |
| **Pluggable Chunking** | Strategy pattern with 3 built-in strategies (paragraph, recursive, semantic) |
| **12-Factor Config** | All settings via environment variables using `pydantic-settings` |
| **Defense in Depth** | Optional API key auth + per-endpoint rate limiting + CORS |
| **Streaming Support** | Real-time SSE token streaming for both `/ask` and `/chat` |

---

## 2. Tech Stack — Deep Dive

### Core Framework

| Technology | Version | Role | Why This Choice |
|---|---|---|---|
| **FastAPI** | ≥0.115.0 | Web framework | Async-first, auto-generates OpenAPI docs, Pydantic-native request validation, dependency injection |
| **Uvicorn** | ≥0.32.0 | ASGI server | Lightning-fast ASGI server built on `uvloop` and `httptools` |
| **Pydantic** | v2 (via FastAPI) | Data validation | Type-safe request/response models with automatic JSON Schema generation |
| **pydantic-settings** | ≥2.7.0 | Configuration | Loads settings from env vars and `.env` files with full type coercion |

### AI / ML Layer

| Technology | Version | Role | Why This Choice |
|---|---|---|---|
| **Ollama** | ≥0.4.0 (Python SDK) | LLM inference | Runs models locally — no API keys, no cloud costs, full privacy. Provides both sync and `AsyncClient` |
| **qwen2.5:0.5b** | — | Chat / generation model | Small (0.5B params), fast inference, good for personal Q&A. Easily swappable via config |
| **nomic-embed-text** | — | Embedding model | 768-dim embeddings, strong performance on retrieval benchmarks, runs locally via Ollama |
| **ChromaDB** | ≥0.6.0 | Vector database | Embedded (no separate server), persistent storage, built-in cosine similarity search, metadata filtering |

### Middleware & Utilities

| Technology | Version | Role | Why This Choice |
|---|---|---|---|
| **sse-starlette** | ≥2.0.0 | Server-Sent Events | Clean SSE integration with FastAPI's `EventSourceResponse` |
| **SlowAPI** | ≥0.1.9 | Rate limiting | Built on `limits` library, integrates directly with FastAPI via decorators |
| **python-multipart** | ≥0.0.18 | File uploads | Required by FastAPI for `UploadFile` / form-data parsing |

### Testing

| Technology | Version | Role |
|---|---|---|
| **pytest** | ≥8.3.0 | Test runner and framework |
| **pytest-cov** | ≥6.0.0 | Code coverage measurement |
| **httpx** | ≥0.28.0 | Async-capable HTTP client (required by FastAPI's `TestClient`) |

### Deployment

| Technology | Role |
|---|---|
| **Docker** | Multi-stage containerization with health check |
| **Python 3.12-slim** | Minimal base image for production |

---

## 3. Project Structure & Module Map

```
rag-api/
├── src/rag_api/                    # Main application package
│   ├── __init__.py                 # Package init, __version__ = "2.0.0"
│   ├── app.py                     # ★ App factory: lifespan, middleware, route registration
│   ├── config.py                  # ★ Centralized Settings (pydantic-settings)
│   ├── database.py                # ★ ChromaDB client, embedding fn, collection accessor
│   ├── models.py                  # ★ All Pydantic request/response schemas
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py              # GET /health, GET /admin/stats
│   │   ├── documents.py           # POST/GET/DELETE /documents
│   │   └── rag.py                 # POST /ask, /ask/stream, /chat, /chat/stream
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunking.py            # Paragraph, recursive, semantic chunking
│   │   ├── retrieval.py           # ChromaDB similarity search
│   │   └── llm.py                 # Ollama async chat + streaming
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py                # Optional API key middleware
│       └── rate_limit.py          # SlowAPI rate limiting
├── tests/                         # Test suite
│   ├── conftest.py                # Shared fixtures (client, cleanup, seeding)
│   ├── test_health.py             # Health & admin stats tests
│   ├── test_documents.py          # Document CRUD tests
│   ├── test_rag.py                # Ask & chat tests
│   ├── test_streaming.py          # SSE streaming tests
│   ├── test_chunking.py           # Chunking strategy unit tests
│   └── test_auth.py               # Auth middleware tests
├── build_knowledge_base.py        # CLI tool for offline text ingestion
├── main.py                        # Legacy v1 monolith (deprecated, kept for reference)
├── profile.txt                    # Sample profile data
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Multi-stage Docker build
├── .env.example                   # Environment variable template
└── config.py                      # Legacy v1 config (root-level, unused by v2)
```

### Module Dependency Graph

```
app.py
  ├── config.py (settings singleton)
  ├── middleware/auth.py (APIKeyMiddleware)
  ├── middleware/rate_limit.py (install_rate_limiter)
  └── routes/
      ├── health.py → database.py, models.py
      ├── documents.py → database.py, models.py, services/chunking.py
      └── rag.py → models.py, services/retrieval.py, services/llm.py
                        │                                   │
                        └─→ database.py                     └─→ config.py
```

---

## 4. Configuration System

**File**: `src/rag_api/config.py`

### How It Works

```python
class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "qwen2.5:0.5b"
    # ... all other settings ...

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

settings = Settings()  # Singleton
```

### Key Design Decisions

1. **`pydantic-settings` over `os.getenv`**: Provides type coercion (e.g., `bool`, `int`, `list[str]`), validation at startup, and auto-documentation via type hints.

2. **Singleton Pattern**: `settings = Settings()` at module level means a single instance is created when the module is first imported and reused everywhere.

3. **`.env` File Support**: In development, settings come from `.env`; in production/Docker, they come from environment variables. `pydantic-settings` handles both transparently.

4. **`case_sensitive = True`**: Environment variable names must match exactly (e.g., `OLLAMA_URL`, not `ollama_url`).

### Settings Breakdown

| Category | Variables | Purpose |
|---|---|---|
| **Ollama** | `OLLAMA_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL` | LLM server connection and model selection |
| **ChromaDB** | `CHROMA_DB_PATH`, `COLLECTION_NAME` | Vector database location and collection name |
| **Retrieval** | `DEFAULT_N_RESULTS` | How many similar chunks to retrieve (default: 3) |
| **Chunking** | `DEFAULT_CHUNK_STRATEGY`, `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` | Document splitting defaults |
| **Rate Limiting** | `RATE_LIMIT_RAG`, `RATE_LIMIT_DOCS` | Per-IP rate limits (e.g., `"60/minute"`) |
| **Auth** | `API_KEY_ENABLED`, `API_KEY` | Toggle and value for API key auth |
| **API** | `APP_TITLE`, `APP_VERSION`, `CORS_ORIGINS` | FastAPI metadata and CORS configuration |
| **Logging** | `LOG_LEVEL` | Python logging level |

### Interview Point — Why Not Just `os.getenv()`?

> `pydantic-settings` gives you **typed** configuration. If `DEFAULT_N_RESULTS` is set to `"abc"` in the environment, the app fails at startup with a clear error instead of failing at runtime with a cryptic `TypeError`. It also auto-generates schema documentation and supports complex types like `list[str]` for CORS origins.

---

## 5. Application Factory

**File**: `src/rag_api/app.py`

### Lifespan (Startup/Shutdown Hook)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ollama.list()  # Verify Ollama is reachable
        logger.info("✅  Connected to Ollama at %s", settings.OLLAMA_URL)
    except Exception as exc:
        logger.warning("⚠️  Ollama not reachable (%s).", exc)
    logger.info("🚀  RAG API v%s ready", settings.APP_VERSION)
    yield  # App runs here
    # (shutdown logic would go after yield)
```

**Why `lifespan` over `@app.on_event("startup")`?**
- `lifespan` is the modern FastAPI pattern (deprecated `on_event` in favor of context manager).
- It allows proper resource cleanup in the `finally`/after-yield section.
- The startup check is **non-blocking** — if Ollama is down, the app still starts but logs a warning.

### Middleware Stack (Order Matters)

```python
# 1. CORS — applied last (outermost), so every response gets CORS headers
app.add_middleware(CORSMiddleware, ...)

# 2. API Key Auth — checked before CORS but after rate limiting
app.add_middleware(APIKeyMiddleware)

# 3. Rate Limiting — innermost, runs first on each request
install_rate_limiter(app)
```

> **Starlette middleware execution order**: Middleware added *last* runs *first* on the request (LIFO). So: **Rate Limiting → Auth → CORS → Route Handler → CORS → Auth → Rate Limiting**.

### Route Registration

```python
app.include_router(health.router)     # /health, /admin/stats
app.include_router(documents.router)  # /documents, /documents/upload, /documents/{user}
app.include_router(rag.router)        # /ask, /ask/stream, /chat, /chat/stream
```

Each router is defined in its own file with `APIRouter` and grouped by `tags` for the OpenAPI docs.

---

## 6. Database Layer

**File**: `src/rag_api/database.py`

### Architecture

```
┌─────────────────────────────────┐
│        database.py              │
│                                 │
│  chroma_client (PersistentClient)│
│         │                       │
│  embedding_fn (OllamaEmbedding) │
│         │                       │
│  get_collection() → Collection  │  ← lru_cache(maxsize=1)
│         │                       │
│  __getattr__("collection") ──┘  │  ← Lazy module-level access
└─────────────────────────────────┘
```

### Key Components

#### 1. `PersistentClient`

```python
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
```

- **Persistent** means data survives process restarts (stored to disk at `./chroma_db/`).
- Alternative: `chromadb.Client()` (in-memory, lost on restart) or `chromadb.HttpClient()` (remote server).
- **Why PersistentClient?** Single-process deployment, no need for a separate ChromaDB server.

#### 2. `OllamaEmbeddingFunction`

```python
embedding_fn = OllamaEmbeddingFunction(
    model_name=settings.OLLAMA_EMBED_MODEL,  # "nomic-embed-text"
    url=settings.OLLAMA_URL,
)
```

- ChromaDB calls this function automatically when you `add()` or `query()` documents.
- Converts text → 768-dimensional float vector via Ollama's embedding endpoint.
- The same embedding function must be used for both indexing and querying (otherwise cosine similarity is meaningless).

#### 3. `get_collection()` with `@lru_cache`

```python
@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    return chroma_client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
```

- `get_or_create_collection`: Creates the collection if it doesn't exist, otherwise returns the existing one.
- `@lru_cache(maxsize=1)`: The collection object is created **once** and reused. Avoids repeated `get_or_create` calls.

#### 4. Lazy `__getattr__` for Backward Compatibility

```python
def __getattr__(name: str):
    if name == "collection":
        return get_collection()
    raise AttributeError(...)
```

- Allows `from src.rag_api.database import collection` (v1 style) to still work.
- The collection is created lazily on first access, not at import time.

### Interview Point — Why Not Create the Collection at Import Time?

> Module-level initialization of ChromaDB would fail during testing if the database file is locked or if we want to mock it. The lazy approach via `lru_cache` + `__getattr__` defers creation until the first actual use, making the module import-safe for tests.

---

## 7. Pydantic Models

**File**: `src/rag_api/models.py`

### Model Hierarchy

```
ChunkStrategy (Enum)
  ├── PARAGRAPH = "paragraph"
  ├── RECURSIVE = "recursive"
  └── SEMANTIC  = "semantic"

DocumentSubmission (Request)
  ├── user_name: str (required, min_length=1)
  ├── content: str (required, min_length=1)
  ├── chunk_strategy: ChunkStrategy (default from config)
  ├── chunk_size: int (50–5000, default from config)
  └── chunk_overlap: int (0–500, default from config)

DocumentOut (Response)
  ├── id: str
  ├── document: str
  └── metadata: dict

AskRequest (Request)
  ├── question: str (required, min_length=1)
  ├── user: Optional[str]
  └── n_results: int (1–20, default from config)

AskResponse (Response)
  ├── question: str
  ├── answer: str
  ├── context_used: list[str]
  └── filtered_by_user: Optional[str]

ChatMessage
  ├── role: str (pattern: user|assistant|system)
  └── content: str

ChatRequest (Request)
  ├── messages: list[ChatMessage] (min_length=1)
  ├── user: Optional[str]
  └── n_results: int (1–20)

ChatResponse (Response)
  ├── reply: str
  ├── context_used: list[str]
  └── filtered_by_user: Optional[str]

HealthResponse
  ├── status: str
  ├── ollama: str
  ├── chromadb: str
  └── collection_count: int

AdminStatsResponse
  ├── total_documents: int
  ├── users: list[str]
  ├── user_document_counts: dict[str, int]
  ├── collection_name: str
  ├── embedding_model: str
  └── chat_model: str
```

### Key Design Decisions

1. **`ChunkStrategy` as `str, Enum`**: Inheriting from both `str` and `Enum` means the value serializes to a plain string in JSON (`"paragraph"` not `"ChunkStrategy.PARAGRAPH"`), which is cleaner for API consumers.

2. **Defaults from `settings`**: Fields like `chunk_size` default to `settings.DEFAULT_CHUNK_SIZE`. This means the `.env` file controls API defaults without code changes.

3. **`ChatMessage.role` with regex pattern**: `pattern="^(user|assistant|system)$"` restricts roles to valid OpenAI-compatible values, catching typos like `"User"` at validation time.

4. **`Field(..., min_length=1)`**: Prevents empty strings from passing validation — `""` and `"   "` are rejected.

### Interview Point — Why Use Pydantic Models Instead of Raw Dicts?

> Pydantic provides: (1) automatic request validation with clear 422 error messages, (2) type coercion (e.g., string `"3"` → int `3`), (3) auto-generated OpenAPI/Swagger schemas, (4) IDE autocompletion and type safety, and (5) serialization control. It eliminates an entire category of bugs where invalid data reaches business logic.

---

## 8. Chunking Service

**File**: `src/rag_api/services/chunking.py`

### Architecture — Strategy Pattern

```
chunk_text(text, strategy="paragraph") ─── dispatcher
    │
    ├── "paragraph" → chunk_by_paragraph(text)
    ├── "recursive" → chunk_recursive(text, chunk_size, chunk_overlap)
    ├── "semantic"  → chunk_semantic(text, chunk_size)
    └── unknown     → ValueError
```

The `chunk_text()` function acts as a **dispatcher** that delegates to the appropriate strategy based on the `strategy` string. This is the **Strategy Pattern** implemented via Python's `match` statement.

### Strategy 1: Paragraph Chunking

```python
def chunk_by_paragraph(text: str) -> list[str]:
    return [c.strip() for c in text.split("\n\n") if c.strip()]
```

- **Algorithm**: Split on double newlines (`\n\n`), strip whitespace, discard empties.
- **Time Complexity**: O(n) where n = length of text.
- **Best For**: Well-structured text with clear paragraph boundaries (e.g., profiles, resumes).
- **Limitation**: No control over chunk size — a 10,000-character paragraph becomes a single chunk.

### Strategy 2: Recursive Character Splitting

```python
def chunk_recursive(text, chunk_size=500, chunk_overlap=50, separators=None):
```

**Algorithm** (3 phases):

1. **Recursive Split** (`_recursive_split`):
   - Try the coarsest separator first (`\n\n` → `\n` → `. ` → ` ` → `""`).
   - If a piece fits within `chunk_size`, keep it.
   - If not, recursively split with the next finer separator.
   - Last resort: hard-cut at `chunk_size` characters.

2. **Merge** (`_merge_with_overlap`):
   - Consecutive small chunks are merged back up to `chunk_size`.
   - Each merged chunk overlaps with the previous by `chunk_overlap` characters.

**Separator Hierarchy**:
```python
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
# Paragraphs → Lines → Sentences → Words → Characters
```

**Why Overlap?** Overlap ensures that information near chunk boundaries isn't lost. If a key sentence spans the boundary between chunk 1 and chunk 2, the overlap ensures it appears in both chunks, so retrieval can find it regardless of which chunk is matched.

**Example**:
```
Input: "AAAA. BBBB. CCCC. DDDD." (chunk_size=15, overlap=5)
After split: ["AAAA.", "BBBB.", "CCCC.", "DDDD."]
After merge: ["AAAA. BBBB.", "BBB. CCCC.", "CCC. DDDD."]
                              ^^^overlap     ^^^overlap
```

### Strategy 3: Semantic Chunking

```python
def chunk_semantic(text, chunk_size=500, similarity_threshold=0.5):
```

**Algorithm**:
1. Split text into sentences using regex: `(?<=[.!?])\s+`
2. Accumulate sentences into a chunk until `chunk_size` is reached.
3. Start a new chunk with the next sentence.

**Current Implementation Note**: This is a *simplified* semantic chunker that groups by sentence boundaries and size, not by actual embedding similarity. A full implementation would:
- Embed each sentence.
- Compare consecutive sentence embeddings (cosine similarity).
- Split where similarity drops below a threshold.

The current approach is still superior to paragraph splitting because it preserves complete sentences.

### Interview Point — Why Multiple Chunking Strategies?

> Different documents have different structures. A well-formatted resume benefits from paragraph chunking; a raw research paper needs recursive splitting; conversational text benefits from sentence-aware semantic chunking. The Strategy Pattern makes it trivial to add new strategies (e.g., token-based chunking, sliding window) without modifying existing code — **Open/Closed Principle**.

---

## 9. Retrieval Service

**File**: `src/rag_api/services/retrieval.py`

### How Retrieval Works

```python
async def retrieve_context(
    question: str,
    user: Optional[str],
    n_results: int,
) -> tuple[list[str], list[dict]]:
```

**Step-by-step**:

1. **Build query parameters**:
   ```python
   query_params = {"query_texts": [question], "n_results": n_results}
   if user:
       query_params["where"] = {"user_name": user}
   ```

2. **Execute similarity search** (offloaded to thread):
   ```python
   results = await asyncio.to_thread(get_collection().query, **query_params)
   ```

3. **Extract results**:
   ```python
   documents = results["documents"][0]  # List of chunk texts
   metadatas = results["metadatas"][0]  # List of metadata dicts
   ```

### What Happens Inside `collection.query()`?

1. The `question` string is passed through `OllamaEmbeddingFunction`, which calls Ollama's `/api/embeddings` endpoint to get a 768-dim vector.
2. ChromaDB computes **cosine similarity** between the query vector and all stored vectors.
3. The top `n_results` most similar chunks are returned, sorted by similarity (highest first).

### User Filtering via Metadata

```python
query_params["where"] = {"user_name": user}
```

ChromaDB's `where` clause applies a **pre-filter** on metadata before the similarity search. This means:
- Only chunks belonging to the specified user are considered.
- The similarity search runs on the filtered subset.
- **Not a post-filter** — this is efficient because ChromaDB filters at the index level.

### Why `asyncio.to_thread()`?

ChromaDB's Python client is **synchronous** (blocking I/O). Calling it directly in an `async` route handler would block the entire FastAPI event loop, preventing other requests from being processed. `asyncio.to_thread()` runs the blocking call in a **thread pool** without blocking the event loop.

### Interview Point — What is Cosine Similarity?

> Cosine similarity measures the angle between two vectors, ignoring magnitude. It's defined as `cos(θ) = (A · B) / (||A|| × ||B||)`. Two texts with similar meaning will have vectors pointing in similar directions, yielding a cosine similarity close to 1.0. Unlike Euclidean distance, cosine similarity is insensitive to vector magnitude, making it ideal for text embeddings where the "direction" of meaning matters more than the "length".

---

## 10. LLM Service

**File**: `src/rag_api/services/llm.py`

### Async Client

```python
_async_client = ollama.AsyncClient(host=settings.OLLAMA_URL)
```

- **Module-level singleton**: Created once, reused for all requests. The client maintains an internal HTTP connection pool to Ollama.
- **`AsyncClient` vs `ollama.chat()`**: The sync `ollama.chat()` would block the event loop. `AsyncClient` uses `httpx` under the hood for non-blocking HTTP.

### Prompt Construction

#### Single-Turn (`build_augmented_prompt`)

```python
def build_augmented_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
```

**Prompt Structure**:
```
System instruction → Context → Question
```

This goes into a single `{"role": "user", "content": prompt}` message. The LLM sees the context and question in one turn.

#### Multi-Turn (`build_system_message`)

```python
def build_system_message(context: str) -> dict:
    return {
        "role": "system",
        "content": "You are a helpful assistant. Use the following retrieved context...\n\n"
                   f"Context:\n{context}",
    }
```

For multi-turn chat, the context is injected as a **system message**, and the full conversation history follows as separate user/assistant messages. This allows the LLM to:
1. Reference the context.
2. Remember prior conversation turns.
3. Maintain conversational coherence.

### Non-Streaming Chat

```python
async def chat(messages: list[dict]) -> str:
    response = await _async_client.chat(
        model=settings.OLLAMA_CHAT_MODEL,
        messages=messages,
    )
    return response["message"]["content"]
```

- Waits for the **complete** response before returning.
- Returns the full answer string.

### Streaming Chat

```python
async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    stream = await _async_client.chat(
        model=settings.OLLAMA_CHAT_MODEL,
        messages=messages,
        stream=True,  # ← Key difference
    )
    async for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        if token:
            yield token
```

- **`stream=True`** tells Ollama to return an async iterator of partial responses.
- Each `chunk` contains a single token (word fragment or word).
- The `yield` makes this function an **async generator**, which the SSE route consumes.

### Error Handling

Both functions catch all exceptions and re-raise as `HTTPException(502)`, indicating an **upstream service failure**. HTTP 502 (Bad Gateway) is semantically correct — the API is acting as a gateway to Ollama, and Ollama failed.

### Interview Point — Why Separate build_augmented_prompt and build_system_message?

> Single-turn and multi-turn RAG use different prompt architectures. In single-turn, everything is packed into one user message because there's no conversation history. In multi-turn, the context must go into the system message so it persists across the full conversation without being repeated in every user turn. This separation makes the prompt engineering testable and swappable independently.

---

## 11. Document Routes

**File**: `src/rag_api/routes/documents.py`

### `POST /documents` — Add via JSON

**Flow**:
```
Request Body → Pydantic Validation → chunk_text() → Generate IDs/Metadata → collection.upsert()
```

**Step-by-step**:
1. Validate request body (`DocumentSubmission` model).
2. Chunk the `content` using the specified strategy via `chunk_text()`.
3. Generate deterministic IDs: `{user_name}-chunk0`, `{user_name}-chunk1`, etc.
4. Build metadata for each chunk: `source`, `user_name`, `chunk_index`, `chunk_strategy`.
5. Upsert to ChromaDB via `asyncio.to_thread()`.

**Why `upsert` instead of `add`?**
- `upsert` = update if ID exists, insert if not.
- If the user re-ingests their profile, chunks are updated rather than duplicated.
- The deterministic ID scheme (`{user_name}-chunk{i}`) makes this work seamlessly.

**Why deterministic IDs?**
- IDs like `default-chunk0` ensure that re-ingestion replaces old chunks.
- Trade-off: if the new content has more/fewer paragraphs than before, chunk indices may shift.

### `POST /documents/upload` — Add via File Upload

**Flow**:
```
.txt file → Read bytes → UTF-8 decode → chunk_text() → Generate IDs/Metadata → collection.upsert()
```

**Validations**:
1. File extension must be `.txt`.
2. File must be valid UTF-8.
3. At least one non-empty chunk must result from parsing.

**Differences from JSON endpoint**:
- Accepts `multipart/form-data` instead of JSON.
- `user_name`, `chunk_strategy`, etc. come from query parameters (not body).
- Metadata includes `filename` and `source: "upload"` instead of `source: "profile"`.

### `GET /documents` — List Chunks

**Query Parameters**:
- `user` (optional): Filter by user name.
- `limit` (1–500, default 50): Max number of chunks to return.

Returns a list of `DocumentOut` objects with `id`, `document` text, and `metadata`.

### `DELETE /documents/{user_name}` — Remove User's Chunks

**Two-step process**:
1. `collection.get(where={"user_name": user_name})` — Find all chunk IDs for this user.
2. `collection.delete(ids=results["ids"])` — Delete by ID.

**Why not `collection.delete(where=...)`?**
- ChromaDB's `delete` supports `where` filters, but the two-step approach gives us the count of deleted documents for the response. It also allows for the 404 check when no documents exist.

---

## 12. RAG Routes

**File**: `src/rag_api/routes/rag.py`

### `POST /ask` — Single-Turn RAG

**Complete flow**:

```
1. Client sends: {"question": "What are Yash's skills?", "user": "default"}
2. Pydantic validates → AskRequest
3. retrieval.retrieve_context(question, user, n_results)
   a. Embed the question via Ollama → 768-dim vector
   b. ChromaDB cosine similarity search → top 3 chunks
4. If no documents found → return early with "No relevant context" message
5. context = "\n\n".join(documents)  # Join retrieved chunks
6. prompt = build_augmented_prompt(context, question)
7. answer = await llm.chat([{"role": "user", "content": prompt}])
8. Return AskResponse with question, answer, context_used, filtered_by_user
```

### `POST /ask/stream` — Streaming Single-Turn RAG

Same as `/ask` but uses Server-Sent Events:

```python
async def _stream():
    yield {"event": "context", "data": json.dumps(documents)}  # Emit context first
    async for token in llm.stream_chat([...]):
        yield {"event": "token", "data": token}                # Stream each token
    yield {"event": "done", "data": ""}                         # Signal completion

return EventSourceResponse(_stream())
```

**Why emit context first?** The client can display the sources/references immediately while the answer streams in token by token.

### `POST /chat` — Multi-Turn Conversational RAG

**Key difference from `/ask`**:

1. **Retrieval query**: Only the **last user message** is used for retrieval (not the full conversation).
   ```python
   last_user_msg = next(
       (m.content for m in reversed(body.messages) if m.role == "user"),
       None,
   )
   ```

2. **Message construction**: The full conversation history is preserved:
   ```python
   ollama_messages = [system_msg] + [
       {"role": m.role, "content": m.content} for m in body.messages
   ]
   ```

3. **Context injection**: Via system message (not inline in user message).

**Why use only the last user message for retrieval?**
> Earlier messages may be about different topics. Using the latest message ensures the retrieved context is relevant to the current question. The LLM still has the full conversation history for coherence.

### `POST /chat/stream` — Streaming Multi-Turn RAG

Combines multi-turn logic with SSE streaming (same event protocol as `/ask/stream`).

---

## 13. Health & Admin Routes

**File**: `src/rag_api/routes/health.py`

### `GET /health`

**Checks**:
1. **Ollama**: Calls `ollama.list()` (lists installed models). If it fails → `"unreachable"`.
2. **ChromaDB**: Calls `collection.count()`. If it fails → `"unreachable"`.
3. **Overall**: `"healthy"` if both are OK, `"degraded"` if either is down.

Both checks run via `asyncio.to_thread()` to avoid blocking.

**Response**:
```json
{
    "status": "healthy",
    "ollama": "ok",
    "chromadb": "ok",
    "collection_count": 42
}
```

### `GET /admin/stats`

Retrieves **all** documents from the collection and computes:
- `total_documents`: Total chunk count.
- `users`: Sorted list of unique user names.
- `user_document_counts`: Dict mapping each user → their chunk count.
- `collection_name`, `embedding_model`, `chat_model`: Current config values.

Uses `collections.Counter` for efficient per-user counting.

**Interview Point — Why is this a Separate Endpoint?**

> `/health` is for monitoring/liveness probes (fast, lightweight). `/admin/stats` fetches all documents and is more expensive — it's for admin dashboards, not health checks. Separating them ensures that Kubernetes liveness probes don't trigger expensive operations.

---

## 14. Authentication Middleware

**File**: `src/rag_api/middleware/auth.py`

### How It Works

```python
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # 1. Auth disabled → pass through
        if not settings.API_KEY_ENABLED:
            return await call_next(request)

        # 2. Public paths → pass through
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # 3. Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != settings.API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})

        return await call_next(request)
```

### Public Paths (Never Require Auth)

```python
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
```

- `/health`: Must be accessible for monitoring/liveness probes.
- `/docs`, `/redoc`, `/openapi.json`: Swagger/ReDoc UI and schema must remain accessible.

### Why Middleware Instead of Dependency Injection?

> Middleware runs **before** the route handler is even matched. This means:
> 1. Unauthenticated requests are rejected *before* any Pydantic validation or database work.
> 2. Every route is automatically protected without adding `Depends(verify_api_key)` to each one.
> 3. New routes are automatically protected.
> Trade-off: It's all-or-nothing (except for the `_PUBLIC_PATHS` whitelist). For fine-grained per-route auth, you'd use FastAPI's dependency injection instead.

### Security Note — Constant-Time Comparison

The current implementation uses `api_key != settings.API_KEY` which is vulnerable to **timing attacks**. A production system should use `hmac.compare_digest()` for constant-time comparison. This is acceptable here because the API key is a secondary defense layer (the primary protection is the local deployment model).

---

## 15. Rate Limiting Middleware

**File**: `src/rag_api/middleware/rate_limit.py`

### How It Works

```python
# 1. Key function — identifies the client by IP address
def _key_func(request: Request) -> str:
    return get_remote_address(request)

# 2. Limiter instance
limiter = Limiter(key_func=_key_func)

# 3. Installation — wires into FastAPI
def install_rate_limiter(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### Usage in Routes

```python
@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.RATE_LIMIT_RAG)  # "60/minute"
async def ask(body: AskRequest, request: Request):
    ...
```

**Note**: The `request: Request` parameter is **required** by SlowAPI even if the route doesn't use it directly. SlowAPI extracts the client IP from it.

### Rate Limit Tiers

| Tier | Default | Endpoints |
|---|---|---|
| **RAG** | `60/minute` | `/ask`, `/ask/stream`, `/chat`, `/chat/stream` |
| **Documents** | `200/minute` | `/documents` (all CRUD operations) |

RAG endpoints have a lower limit because each call triggers an embedding + LLM inference, which is computationally expensive.

### What Happens When Exceeded?

SlowAPI returns HTTP 429 (Too Many Requests) with a `Retry-After` header indicating when the client can retry.

---

## 16. CLI Ingestion Tool

**File**: `build_knowledge_base.py`

### Purpose

Offline CLI tool to bulk-ingest text files into ChromaDB, independent of the web server.

### Usage

```bash
python build_knowledge_base.py                                     # defaults
python build_knowledge_base.py -f resume.txt -u alice              # custom file & user
python build_knowledge_base.py --strategy recursive --chunk-size 300
python build_knowledge_base.py --force                             # overwrite existing
```

### Flow

```
parse_args() → load_file() → chunk_text() → ChromaDB connect → Duplicate check → upsert()
```

### Duplicate Detection

```python
if not args.force:
    existing = collection.get(ids=ids)
    already = [eid for eid in existing["ids"] if eid]
    if already:
        # Filter out existing chunks
        new_pairs = [(cid, chunk) for cid, chunk in zip(ids, chunks) if cid not in already]
```

- By default, skips chunks that already exist (matched by deterministic ID).
- `--force` flag bypasses this check and overwrites everything.

### Why a Separate CLI Tool?

> The API's `POST /documents` endpoint requires the server to be running. The CLI tool can be used in:
> 1. **Initial setup**: Seed the database before the first server start.
> 2. **Batch operations**: Ingest multiple files without HTTP overhead.
> 3. **CI/CD pipelines**: Automate knowledge base updates in deployment scripts.

---

## 17. Docker & Deployment

**File**: `Dockerfile`

### Multi-Stage Build

```dockerfile
# Stage 1: Build — install dependencies with build tools
FROM python:3.12-slim AS base
RUN apt-get install -y build-essential  # Needed by chromadb (compiles SQLite)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — copy only what's needed
FROM python:3.12-slim
COPY --from=base /usr/local/lib/python3.12/site-packages ...
COPY --from=base /usr/local/bin ...
COPY . .
```

**Why Multi-Stage?**
- Stage 1 has `build-essential` (gcc, make, etc.) — ~200MB+ of build tools.
- Stage 2 only has the compiled packages — no build tools in the final image.
- **Result**: Smaller image (~300MB vs ~500MB+), smaller attack surface.

### Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

- Docker checks `/health` every 30 seconds.
- After 3 consecutive failures, the container is marked `unhealthy`.
- Orchestrators (Docker Compose, Kubernetes) can auto-restart unhealthy containers.

### Volume Mount

```dockerfile
VOLUME ["/app/chroma_db"]
```

ChromaDB data is persisted in a Docker volume, surviving container restarts and updates.

### Running with Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 \
    -e OLLAMA_URL=http://host.docker.internal:11434 \
    -v rag-data:/app/chroma_db \
    rag-api
```

**Note**: `host.docker.internal` resolves to the host machine's IP, allowing the container to reach Ollama running on the host.

---

## 18. Testing Strategy

### Test Architecture

```
tests/
├── conftest.py          # Shared fixtures
├── test_health.py       # 4 tests — health + admin stats
├── test_documents.py    # 8 tests — CRUD + upload + strategies
├── test_rag.py          # 5 tests — ask + chat
├── test_streaming.py    # 4 tests — SSE streaming
├── test_chunking.py     # 12 tests — unit tests for all strategies
└── test_auth.py         # 5 tests — auth enabled/disabled
```

### Key Fixtures (`conftest.py`)

```python
# Patch ollama.list to prevent real network calls during import
with patch("ollama.list", return_value=MagicMock()):
    from src.rag_api.app import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Remove test documents after each test
    collection.get(where={"user_name": TEST_USER})
    collection.delete(ids=results["ids"])

@pytest.fixture
def seeded_docs(client):
    # Pre-populate the database with test data
    client.post("/documents", json={...})
    return client
```

**Key decisions**:
1. **`autouse=True` on cleanup**: Every test automatically cleans up its data. Tests are isolated.
2. **`seeded_docs` fixture**: Tests that need pre-existing data use this fixture (dependency injection).
3. **Ollama mock at import time**: The app's lifespan calls `ollama.list()` on startup. Patching before import prevents test hangs.

### Mocking Strategy

- **Ollama LLM calls**: Mocked via `@patch("src.rag_api.services.llm._async_client")` — avoids needing a running Ollama server for tests.
- **ChromaDB**: **Not mocked** — tests use a real ChromaDB instance. This provides higher confidence but requires the `chroma_db/` directory to be writable.
- **SSE Streaming**: The mock returns an async iterator to simulate token-by-token streaming.

### Running Tests

```bash
pytest tests/ -v                                            # Run all tests
pytest tests/ -v --cov=src/rag_api --cov-report=term-missing  # With coverage
pytest tests/test_chunking.py -v                            # Single file
```

---

## 19. SSE Streaming Protocol

### Event Types

| Event | Data | When |
|---|---|---|
| `context` | JSON array of strings | First event — retrieved context documents |
| `token` | Raw text token | Per-token — each word/fragment from the LLM |
| `done` | Empty string | Last event — signals stream completion |

### Wire Format

```
event: context
data: ["Alice is a software engineer at Acme Corp.", "She specializes in distributed systems."]

event: token
data: Alice

event: token
data:  is

event: token
data:  a

event: token
data:  software

event: token
data:  engineer

event: done
data:
```

### Client Consumption (JavaScript)

```javascript
const eventSource = new EventSource('/ask/stream', {method: 'POST', body: ...});

eventSource.addEventListener('context', (e) => {
    const docs = JSON.parse(e.data);
    displaySources(docs);
});

eventSource.addEventListener('token', (e) => {
    appendToAnswer(e.data);
});

eventSource.addEventListener('done', () => {
    eventSource.close();
});
```

### Why SSE Over WebSockets?

> 1. **Simpler**: SSE is unidirectional (server → client), which matches the LLM streaming use case.
> 2. **HTTP-native**: Works over regular HTTP, no upgrade handshake needed.
> 3. **Auto-reconnect**: Browsers automatically reconnect on SSE disconnections.
> 4. **Firewall-friendly**: SSE uses standard HTTP — no issues with proxies or firewalls.
> WebSockets would be overkill since the client only sends one request and receives a stream back.

---

## 20. End-to-End Data Flow

### Ingestion Flow

```
profile.txt
    │
    ▼
build_knowledge_base.py  (or POST /documents)
    │
    ▼
chunk_text(strategy="paragraph")
    │ Split text → ["chunk0", "chunk1", "chunk2", "chunk3"]
    ▼
OllamaEmbeddingFunction
    │ Each chunk → 768-dim vector via Ollama /api/embeddings
    ▼
ChromaDB.upsert()
    │ Store vectors + text + metadata on disk
    ▼
chroma_db/ directory (SQLite + parquet files)
```

### Query Flow (Single-Turn)

```
Client: POST /ask {"question": "What are Yash's skills?", "user": "default"}
    │
    ▼
Middleware Stack: Rate Limit → Auth Check → CORS
    │
    ▼
Pydantic Validation → AskRequest
    │
    ▼
retrieval.retrieve_context("What are Yash's skills?", "default", 3)
    │
    ├── Embed the question → [0.12, -0.45, 0.78, ...] (768 dims)
    │
    ├── ChromaDB similarity search (cosine) with where={"user_name": "default"}
    │
    └── Return top 3 chunks:
        ["He has experience with Python, JavaScript, AWS, and Docker.",
         "He's currently learning about cloud computing, AI, and DevOps.",
         "His career goal is to become a DevOps engineer..."]
    │
    ▼
llm.build_augmented_prompt(context, question)
    │ "You are a helpful assistant. Use the following context...
    │  Context: He has experience with Python... He's currently learning...
    │  Question: What are Yash's skills?"
    ▼
llm.chat([{"role": "user", "content": prompt}])
    │
    ├── Ollama /api/chat (model: qwen2.5:0.5b)
    │
    └── "Yash has experience with Python, JavaScript, AWS, and Docker.
         He is currently learning cloud computing, AI, and DevOps."
    │
    ▼
AskResponse
    │ {"question": "...", "answer": "...", "context_used": [...], "filtered_by_user": "default"}
    ▼
Client receives JSON response
```

### Query Flow (Multi-Turn Chat)

```
Client: POST /chat {
    "messages": [
        {"role": "user", "content": "Tell me about Yash."},
        {"role": "assistant", "content": "Yash is interested in cloud computing and AI."},
        {"role": "user", "content": "What technologies does he know?"}
    ],
    "user": "default"
}
    │
    ▼
Extract last user message: "What technologies does he know?"
    │
    ▼
retrieval.retrieve_context("What technologies does he know?", ...)
    │ → ["He has experience with Python, JavaScript, AWS, and Docker."]
    ▼
Build message list for Ollama:
    [
        {"role": "system", "content": "You are a helpful assistant. Context:\n..."},
        {"role": "user", "content": "Tell me about Yash."},
        {"role": "assistant", "content": "Yash is interested in cloud computing and AI."},
        {"role": "user", "content": "What technologies does he know?"}
    ]
    │
    ▼
llm.chat(ollama_messages) → "He knows Python, JavaScript, AWS, and Docker."
```

---

## 21. Interview Q&A Bank

### Architecture & Design

**Q: What is RAG and why did you choose this architecture?**
> RAG (Retrieval-Augmented Generation) combines a retriever (vector database) with a generator (LLM). I chose it because LLMs have a knowledge cutoff and can't access private data. Instead of fine-tuning (expensive, inflexible), RAG retrieves relevant documents at query time and injects them into the prompt. This makes the system dynamic — updating the knowledge base doesn't require retraining the model.

**Q: Why did you choose FastAPI over Flask or Django?**
> FastAPI is async-first (critical for non-blocking LLM calls), has native Pydantic integration for request validation, auto-generates OpenAPI docs, and has built-in dependency injection. Flask would require adding these features manually. Django is too heavyweight for a focused API service.

**Q: Explain the middleware execution order.**
> Starlette middleware uses a LIFO stack. The last middleware added is the first to execute on incoming requests. In our stack: Rate Limiting runs first (rejects over-limit requests cheaply), then Auth checks the API key, then CORS adds headers. On the response path, they execute in reverse.

**Q: How does your system handle concurrent requests?**
> FastAPI runs on an ASGI server (Uvicorn) with an async event loop. Blocking operations (ChromaDB, Ollama sync client) are offloaded to a thread pool via `asyncio.to_thread()`. The Ollama `AsyncClient` uses native async HTTP. This means the event loop is never blocked, and the server can handle many concurrent requests with a single process.

### Data & Storage

**Q: Why ChromaDB over Pinecone or Weaviate?**
> ChromaDB is embedded (no separate server to manage), persistent (survives restarts), and has a simple Python API. For a single-server deployment with local LLM inference, there's no need for a distributed vector database. If we needed horizontal scaling, Pinecone or Weaviate would be better choices.

**Q: How do embeddings work in your system?**
> Each document chunk is converted to a 768-dimensional float vector by the `nomic-embed-text` model running on Ollama. Semantically similar texts produce vectors that are close in cosine distance. At query time, the question is also embedded, and ChromaDB finds the chunks whose vectors are most similar to the question vector.

**Q: What happens if two users upload documents with the same content?**
> Each user's chunks have unique IDs (`{user_name}-chunk{i}`). Two users uploading identical content get separate entries in ChromaDB. The metadata `where` filter ensures queries are scoped to the correct user. There's no cross-user data leakage.

### Performance & Scalability

**Q: What are the performance bottlenecks?**
> 1. **Embedding generation** (~50ms per chunk via Ollama) — dominates ingestion time.
> 2. **LLM inference** (~200ms–2s per query depending on model and response length) — dominates query time.
> 3. **ChromaDB similarity search** (~5-10ms for <10K documents) — negligible.
> The rate limiter prevents any single client from monopolizing LLM resources.

**Q: How would you scale this system?**
> 1. **Vertical**: Use a larger Ollama model (better answers) or GPU acceleration (faster inference).
> 2. **Horizontal**: Replace ChromaDB with a distributed vector DB (Pinecone/Qdrant), run multiple API instances behind a load balancer, and connect them to a shared Ollama cluster.
> 3. **Caching**: Add Redis caching for frequently asked questions (same question + context = same answer).

### Security

**Q: How does your authentication work?**
> It's an opt-in API key system. When `API_KEY_ENABLED=true`, every request (except `/health` and docs) must include an `X-API-Key` header. The middleware checks this before the request reaches any route handler. Public endpoints like `/health` are whitelisted for monitoring tools.

**Q: What are the security considerations you'd address for production?**
> 1. Use `hmac.compare_digest()` for timing-attack-safe key comparison.
> 2. Add HTTPS (TLS termination at the reverse proxy level).
> 3. Implement per-user API keys (not a single shared key).
> 4. Add input sanitization for prompt injection attacks.
> 5. Set restrictive CORS origins instead of `["*"]`.
> 6. Add request logging and audit trails.

### Testing

**Q: How do you test without a running Ollama server?**
> Ollama LLM calls are mocked at the `_async_client` level using `unittest.mock.patch`. The mock returns predetermined responses (`AsyncMock(return_value={"message": {"content": "Mocked answer."}})`). ChromaDB is NOT mocked — tests use a real embedded ChromaDB instance for higher confidence. An `autouse` cleanup fixture ensures test isolation by deleting test data after each test.

**Q: How do you test SSE streaming endpoints?**
> The streaming mock returns an async generator that yields predetermined chunks. We verify: (1) the response content-type is `text/event-stream`, (2) the status code is 200, (3) edge cases like no-context and missing-user-message are handled. A custom `_parse_sse_events()` helper parses raw SSE text into structured event/data dicts for assertions.

---

## Appendix: Environment Variables Quick Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `OLLAMA_URL` | `str` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | `str` | `qwen2.5:0.5b` | LLM for chat |
| `OLLAMA_EMBED_MODEL` | `str` | `nomic-embed-text` | Model for embeddings |
| `CHROMA_DB_PATH` | `str` | `./chroma_db` | Vector DB storage path |
| `COLLECTION_NAME` | `str` | `personal_profile` | ChromaDB collection |
| `DEFAULT_N_RESULTS` | `int` | `3` | Number of chunks to retrieve |
| `DEFAULT_CHUNK_STRATEGY` | `str` | `paragraph` | Default chunking strategy |
| `DEFAULT_CHUNK_SIZE` | `int` | `500` | Target chunk size (chars) |
| `DEFAULT_CHUNK_OVERLAP` | `int` | `50` | Overlap between chunks |
| `RATE_LIMIT_RAG` | `str` | `60/minute` | Rate limit for RAG endpoints |
| `RATE_LIMIT_DOCS` | `str` | `200/minute` | Rate limit for document endpoints |
| `API_KEY_ENABLED` | `bool` | `false` | Enable API key auth |
| `API_KEY` | `str` | `""` | The API key value |
| `CORS_ORIGINS` | `list[str]` | `["*"]` | Allowed CORS origins |
| `LOG_LEVEL` | `str` | `INFO` | Python logging level |
