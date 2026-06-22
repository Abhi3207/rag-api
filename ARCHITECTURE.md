# 🏗️ RAG API — Architecture Document

> **Purpose**: A deep-dive into the system architecture, design patterns, component interactions, and deployment topology. This document complements `IMPLEMENTATION.md` and is designed for technical interviews and architecture discussions.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component Architecture](#3-component-architecture)
4. [Layered Architecture](#4-layered-architecture)
5. [Data Architecture](#5-data-architecture)
6. [Request Lifecycle](#6-request-lifecycle)
7. [Concurrency Model](#7-concurrency-model)
8. [Design Patterns Used](#8-design-patterns-used)
9. [API Architecture](#9-api-architecture)
10. [Embedding & Vector Search Pipeline](#10-embedding--vector-search-pipeline)
11. [Prompt Engineering Architecture](#11-prompt-engineering-architecture)
12. [Middleware Pipeline](#12-middleware-pipeline)
13. [Error Handling Architecture](#13-error-handling-architecture)
14. [Configuration Architecture](#14-configuration-architecture)
15. [Testing Architecture](#15-testing-architecture)
16. [Deployment Architecture](#16-deployment-architecture)
17. [Security Architecture](#17-security-architecture)
18. [Scalability & Evolution Roadmap](#18-scalability--evolution-roadmap)
19. [Architecture Decision Records (ADRs)](#19-architecture-decision-records-adrs)
20. [Interview Deep-Dive Questions](#20-interview-deep-dive-questions)

---

## 1. System Architecture Overview

The RAG API follows a **layered, modular architecture** with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  REST clients, cURL, Swagger UI, Frontend apps                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼─────────────────────────────────────┐
│                     MIDDLEWARE LAYER                              │
│  CORS → API Key Auth → Rate Limiting                             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      ROUTING LAYER                               │
│  health.py │ documents.py │ rag.py                               │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     SERVICE LAYER                                │
│  chunking.py │ retrieval.py │ llm.py                             │
└───────────┬────────────────┬────────────────┬────────────────────┘
            │                │                │
┌───────────▼──┐  ┌──────────▼──────┐  ┌──────▼──────┐
│  Pure Logic  │  │    ChromaDB     │  │   Ollama    │
│  (chunking)  │  │  (Vector DB)    │  │   (LLM)     │
└──────────────┘  └─────────────────┘  └─────────────┘
```

### Core Principles

| Principle | Description |
|---|---|
| **Separation of Concerns** | Each layer has a single responsibility: routing, business logic, data access |
| **Dependency Inversion** | Routes depend on service abstractions, not concrete implementations |
| **Single Responsibility** | Each module handles one concern (auth, rate limiting, chunking, etc.) |
| **Configuration Externalization** | All settings live in environment variables, not in code |
| **Async-First** | Every I/O operation is non-blocking |

---

## 2. High-Level Architecture Diagram

### System Context

```
                    ┌─────────────────┐
                    │   Developer /   │
                    │   End User      │
                    └────────┬────────┘
                             │
                     HTTP/SSE Requests
                             │
                    ┌────────▼────────┐
                    │                 │
                    │    RAG API      │
                    │   (FastAPI)     │
                    │                 │
                    │  Port 8000      │
                    └───┬─────────┬───┘
                        │         │
              ┌─────────▼──┐   ┌──▼──────────┐
              │  ChromaDB   │   │   Ollama     │
              │ (embedded)  │   │  (localhost)  │
              │             │   │              │
              │ ./chroma_db │   │ Port 11434   │
              └─────────────┘   └──────────────┘
                                      │
                              ┌───────┴────────┐
                              │  Local Models   │
                              │                 │
                              │ qwen2.5:0.5b    │
                              │ nomic-embed-text│
                              └────────────────┘
```

### Request Flow Architecture

```
Request ──► CORS ──► Auth ──► Rate Limit ──► Router ──► Service ──► External
                                                │              │
                                                ▼              ▼
                                         Pydantic Model    ChromaDB / Ollama
                                         Validation
```

---

## 3. Component Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       src/rag_api/                            │
│                                                              │
│  ┌────────────┐    ┌──────────────────────────────────────┐  │
│  │  app.py    │    │           routes/                     │  │
│  │            │───►│  ┌──────────┐ ┌────────┐ ┌────────┐ │  │
│  │  (Factory) │    │  │health.py │ │docs.py │ │ rag.py │ │  │
│  └─────┬──────┘    │  └────┬─────┘ └───┬────┘ └───┬────┘ │  │
│        │           └───────┼───────────┼──────────┼──────┘  │
│        │                   │           │          │          │
│  ┌─────▼──────┐    ┌──────▼───────────▼──────────▼───────┐  │
│  │ middleware/ │    │          services/                   │  │
│  │            │    │  ┌──────────┐ ┌──────┐ ┌──────────┐ │  │
│  │ ┌────────┐ │    │  │chunking  │ │ llm  │ │retrieval │ │  │
│  │ │ auth   │ │    │  │ .py     │ │ .py  │ │  .py     │ │  │
│  │ └────────┘ │    │  └──────────┘ └──┬───┘ └────┬─────┘ │  │
│  │ ┌────────┐ │    └──────────────────┼──────────┼───────┘  │
│  │ │rate_   │ │                       │          │          │
│  │ │limit   │ │    ┌──────────────────▼──────────▼───────┐  │
│  │ └────────┘ │    │          Shared Modules              │  │
│  └────────────┘    │  ┌──────────┐ ┌──────────┐          │  │
│                    │  │config.py │ │database.py│          │  │
│  ┌────────────┐    │  └──────────┘ └──────────┘          │  │
│  │ models.py  │    └─────────────────────────────────────┘  │
│  └────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Depends On |
|---|---|---|
| `app.py` | App creation, middleware registration, router inclusion | config, middleware/*, routes/* |
| `config.py` | Centralized typed configuration | (none — leaf node) |
| `database.py` | ChromaDB client, embedding function, collection accessor | config |
| `models.py` | All Pydantic request/response schemas | config |
| `routes/health.py` | System health & admin analytics | database, models |
| `routes/documents.py` | Document CRUD operations | database, models, services/chunking |
| `routes/rag.py` | RAG query endpoints (ask, chat, streaming) | models, services/retrieval, services/llm |
| `services/chunking.py` | Text chunking algorithms | (none — pure logic) |
| `services/retrieval.py` | Vector similarity search | database |
| `services/llm.py` | LLM inference (sync & streaming) | config |
| `middleware/auth.py` | API key authentication | config |
| `middleware/rate_limit.py` | Per-IP rate limiting | (none) |

---

## 4. Layered Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────┐
│ PRESENTATION LAYER                           │
│ • FastAPI route handlers                     │
│ • Pydantic request/response serialization    │
│ • SSE EventSourceResponse                    │
│ • HTTP status codes                          │
├─────────────────────────────────────────────┤
│ BUSINESS LOGIC LAYER (Services)              │
│ • Chunking algorithms (pure functions)       │
│ • Prompt construction (template logic)       │
│ • Retrieval orchestration                    │
├─────────────────────────────────────────────┤
│ DATA ACCESS LAYER                            │
│ • ChromaDB client (PersistentClient)         │
│ • Ollama AsyncClient                         │
│ • Embedding function                         │
├─────────────────────────────────────────────┤
│ INFRASTRUCTURE LAYER                         │
│ • Disk storage (chroma_db/)                  │
│ • Ollama server (localhost:11434)             │
│ • Docker, environment variables              │
└─────────────────────────────────────────────┘
```

### Layer Rules

1. **Top layers depend on bottom layers**, never the reverse.
2. **Services never import from routes** — services don't know about HTTP.
3. **Database layer doesn't know about business rules** — it just stores and retrieves.
4. **Config is a cross-cutting concern** — imported by all layers except pure logic.

---

## 5. Data Architecture

### Data Model

```
ChromaDB Collection: "personal_profile"
┌──────────────────────────────────────────────────────┐
│                                                      │
│  Document Record                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │ id: "default-chunk0"                           │  │
│  │ document: "My name is Yash. I'm currently..."  │  │
│  │ embedding: [0.12, -0.45, 0.78, ..., 0.33]     │  │
│  │            └── 768 floats (nomic-embed-text) ──┘  │
│  │ metadata: {                                    │  │
│  │   "source": "profile",                         │  │
│  │   "user_name": "default",                      │  │
│  │   "chunk_index": 0,                            │  │
│  │   "chunk_strategy": "paragraph"                │  │
│  │ }                                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  Index: HNSW (Hierarchical Navigable Small World)    │
│  Distance: Cosine Similarity                         │
│  Storage: SQLite + Parquet files on disk              │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Data Flow — Ingestion Pipeline

```
Raw Text
    │
    ▼
┌───────────────────┐
│  Text Chunking    │  chunk_text(strategy, chunk_size, overlap)
│                   │
│  "paragraph"  ─── split on \n\n
│  "recursive"  ─── hierarchical splitting + overlap merging
│  "semantic"   ─── sentence-aware grouping
└───────┬───────────┘
        │
        ▼ List[str] (chunks)
┌───────────────────┐
│  ID Generation    │  f"{user_name}-chunk{i}"
│                   │  Deterministic → enables upsert
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Metadata Build   │  {source, user_name, chunk_index, strategy}
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  ChromaDB Upsert  │  Embedding is generated automatically
│                   │  by OllamaEmbeddingFunction
└───────────────────┘
```

### Data Flow — Query Pipeline

```
Question String
    │
    ▼
┌───────────────────┐
│  Embedding        │  OllamaEmbeddingFunction → 768-dim vector
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  HNSW Index       │  Approximate Nearest Neighbor search
│  Search           │  Filtered by metadata (user_name)
│                   │  Returns top-k by cosine similarity
└───────┬───────────┘
        │
        ▼ List[str] (top-k documents)
┌───────────────────┐
│  Context Assembly │  "\n\n".join(documents)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Prompt Build     │  Inject context into LLM prompt template
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  LLM Inference    │  Ollama chat API → generated answer
└───────────────────┘
```

### Storage Architecture

```
chroma_db/                    # ChromaDB persistence directory
├── chroma.sqlite3            # Metadata, collection info, document text
└── {collection_uuid}/
    ├── data_level0.bin       # HNSW index (vector data)
    ├── header.bin            # Index metadata
    ├── length.bin            # Vector lengths
    └── link_lists.bin        # HNSW graph structure
```

**ChromaDB uses two storage backends**:
1. **SQLite**: Stores metadata, document text, and collection schemas.
2. **HNSW index files**: Binary files storing the vector index for fast ANN search.

---

## 6. Request Lifecycle

### Complete Request Lifecycle — `POST /ask`

```
1. CLIENT
   └── HTTP POST /ask {"question": "What are Yash's skills?", "user": "default"}

2. UVICORN (ASGI Server)
   └── Receives raw HTTP → creates ASGI scope → passes to FastAPI

3. CORS MIDDLEWARE (outermost)
   └── Adds Access-Control-* headers to response
   └── Passes request inward

4. API KEY MIDDLEWARE
   ├── settings.API_KEY_ENABLED == false → pass through
   └── (if enabled: check X-API-Key header)

5. RATE LIMIT CHECK
   ├── Extract client IP via get_remote_address()
   ├── Check against "60/minute" limit
   ├── Under limit → pass through
   └── Over limit → return 429 Too Many Requests

6. FASTAPI ROUTING
   └── Match POST /ask → rag.router → ask() handler

7. PYDANTIC VALIDATION
   ├── Parse JSON body → AskRequest model
   ├── Validate: question not empty, n_results in range
   └── Validation failure → return 422 Unprocessable Entity

8. RETRIEVAL SERVICE (async)
   ├── Build query params: query_texts, n_results, where filter
   ├── asyncio.to_thread(collection.query, ...)
   │   ├── OllamaEmbeddingFunction embeds the question
   │   ├── HNSW index cosine similarity search
   │   └── Return top-k documents + metadatas
   └── Return (documents, metadatas)

9. PROMPT CONSTRUCTION
   └── build_augmented_prompt(context, question)

10. LLM INFERENCE (async)
    ├── _async_client.chat(model, messages)
    │   ├── HTTP POST to Ollama /api/chat
    │   ├── Ollama runs inference on qwen2.5:0.5b
    │   └── Returns complete response
    └── Extract message.content

11. RESPONSE SERIALIZATION
    └── AskResponse → JSON

12. MIDDLEWARE RESPONSE PATH
    └── CORS headers added → response sent to client
```

---

## 7. Concurrency Model

### Async Architecture

```
┌──────────────────────────────────────────────────┐
│                UVICORN PROCESS                    │
│                                                  │
│  ┌───────────────────────────────────────────┐   │
│  │           ASYNCIO EVENT LOOP               │   │
│  │                                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │Request 1│  │Request 2│  │Request 3│  │   │
│  │  │(ask)    │  │(chat)   │  │(health) │  │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  │   │
│  │       │            │            │         │   │
│  │       ▼            ▼            ▼         │   │
│  │  ┌─────────────────────────────────────┐  │   │
│  │  │         THREAD POOL EXECUTOR        │  │   │
│  │  │  (for blocking ChromaDB calls)      │  │   │
│  │  │                                     │  │   │
│  │  │  Thread 1: collection.query(...)    │  │   │
│  │  │  Thread 2: collection.get(...)      │  │   │
│  │  │  Thread 3: ollama.list()            │  │   │
│  │  └─────────────────────────────────────┘  │   │
│  │                                           │   │
│  │  ┌─────────────────────────────────────┐  │   │
│  │  │      ASYNC HTTP CONNECTIONS         │  │   │
│  │  │  (for Ollama AsyncClient)           │  │   │
│  │  │                                     │  │   │
│  │  │  Conn 1: POST /api/chat (stream)    │  │   │
│  │  │  Conn 2: POST /api/chat             │  │   │
│  │  └─────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Two Patterns for Async I/O

| Pattern | Used For | Example |
|---|---|---|
| `asyncio.to_thread()` | Wrapping synchronous blocking calls | `ChromaDB.query()`, `ChromaDB.count()`, `ollama.list()` |
| Native `async/await` | Natively async libraries | `ollama.AsyncClient.chat()` |

### Why Not Make ChromaDB Fully Async?

ChromaDB's Python client is built on synchronous SQLite and file I/O. While ChromaDB offers an `HttpClient` for remote access (which could be async), the `PersistentClient` used here is inherently synchronous. `asyncio.to_thread()` is the standard pattern for integrating sync libraries into async frameworks — it runs the blocking call on a worker thread from Python's default `ThreadPoolExecutor`.

---

## 8. Design Patterns Used

### 1. Strategy Pattern — Chunking

```
               chunk_text() ← Dispatcher
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   paragraph    recursive    semantic
```

**Pattern**: Define a family of algorithms (chunking strategies), encapsulate each one in a function, and make them interchangeable via a dispatcher.

**Benefit**: Adding a new strategy (e.g., `"token"` or `"sliding_window"`) requires only:
1. Writing the new function.
2. Adding one `case` to the `match` statement.
No existing code changes needed — **Open/Closed Principle**.

### 2. Singleton Pattern — Configuration

```python
# config.py
settings = Settings()  # Created once at module level

# Used everywhere:
from src.rag_api.config import settings
```

Python's module system naturally supports singletons — importing the same module multiple times reuses the same module object. `settings` is created once and shared.

### 3. Factory Pattern — App Creation

```python
# app.py
app = FastAPI(title=..., version=..., lifespan=...)
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(APIKeyMiddleware)
install_rate_limiter(app)
app.include_router(health.router)
...
```

The `app.py` module acts as a factory that assembles the application from its components (middleware, routes, config).

### 4. Repository Pattern — Database Layer

```python
# database.py
get_collection()  → provides the data access object
# services/retrieval.py
retrieve_context(question, user, n_results)  → abstracts query logic
```

The database layer provides a clean interface (`get_collection()`) that hides ChromaDB internals. The retrieval service further abstracts this into a business-level operation.

### 5. Middleware Chain — Request Processing

```
Request → CORS → Auth → RateLimit → Router → Response
```

Classic **Chain of Responsibility** pattern where each middleware either handles the request (e.g., returns 401) or passes it to the next handler.

### 6. Template Method — Prompt Construction

```python
build_augmented_prompt(context, question)  # Template for single-turn
build_system_message(context)              # Template for multi-turn
```

Prompt templates define the structure; the variable parts (context, question) are injected at runtime.

### 7. Lazy Initialization — Collection Access

```python
@lru_cache(maxsize=1)
def get_collection():
    return chroma_client.get_or_create_collection(...)
```

The collection is created on first access, not at import time. This avoids import-time side effects and makes testing easier.

---

## 9. API Architecture

### RESTful Design

```
/health              GET     → System health check
/admin/stats         GET     → Knowledge base analytics

/documents           POST    → Create (add chunks)
/documents           GET     → Read (list chunks)
/documents/upload    POST    → Create (via file upload)
/documents/{user}    DELETE  → Delete (user's chunks)

/ask                 POST    → Single-turn RAG query
/ask/stream          POST    → Single-turn RAG (SSE)
/chat                POST    → Multi-turn RAG query
/chat/stream         POST    → Multi-turn RAG (SSE)
```

### API Versioning Strategy

Currently **URL-based implicit versioning** (no `/v1/` prefix). The `APP_VERSION` setting (`2.0.0`) tracks the overall version. For breaking changes, the recommended approach is:
1. Add a `/v2/` prefix to new routes.
2. Keep old routes active with deprecation warnings.
3. Remove after a migration period.

### Response Consistency

All error responses follow FastAPI's standard format:
```json
{"detail": "Error message here"}
```

Success responses use typed Pydantic models for consistency and documentation.

### Content Types

| Endpoint | Request | Response |
|---|---|---|
| `/ask`, `/chat` | `application/json` | `application/json` |
| `/ask/stream`, `/chat/stream` | `application/json` | `text/event-stream` |
| `/documents/upload` | `multipart/form-data` | `application/json` |
| `/documents` (POST) | `application/json` | `application/json` |

---

## 10. Embedding & Vector Search Pipeline

### Embedding Architecture

```
                    nomic-embed-text
Input Text ─────────────────────────────► 768-dim Float Vector

"What are Yash's skills?"  ──►  [0.123, -0.456, 0.789, ..., 0.321]
                                 └─────── 768 dimensions ──────────┘
```

### How the Embedding Model Works

1. **Tokenization**: The input text is split into subword tokens.
2. **Transformer Encoding**: Tokens pass through transformer layers (attention + feedforward).
3. **Pooling**: Token representations are aggregated (typically mean pooling) into a single vector.
4. **Normalization**: The vector is L2-normalized so cosine similarity = dot product.

### HNSW Index Architecture

ChromaDB uses **HNSW** (Hierarchical Navigable Small World) for approximate nearest neighbor search:

```
Layer 2:  [A] ─────────────────── [F]
           │                       │
Layer 1:  [A] ── [C] ──── [E] ── [F]
           │      │        │      │
Layer 0:  [A]-[B]-[C]-[D]-[E]-[F]-[G]-[H]
```

**How HNSW works**:
1. **Multi-layer graph**: Upper layers have fewer nodes (for coarse navigation), lower layers have all nodes.
2. **Search**: Start at the top layer, greedily move toward the query vector, drop to the next layer, repeat.
3. **Time Complexity**: O(log N) average search time vs O(N) for brute-force.
4. **Trade-off**: Uses more memory than flat indices, but dramatically faster for >1000 documents.

### Cosine Similarity Calculation

```
similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
  A · B = Σ(Ai × Bi)           (dot product)
  ||A|| = √(Σ(Ai²))            (L2 norm)

Range: [-1, 1]
  1.0  = identical direction (maximum similarity)
  0.0  = orthogonal (no similarity)
 -1.0  = opposite direction (maximum dissimilarity)
```

### Why Cosine Over Euclidean Distance?

- **Scale invariant**: Cosine measures direction, not magnitude. Two documents about "Python programming" will have similar vectors regardless of document length.
- **Normalized**: nomic-embed-text produces L2-normalized vectors, so cosine similarity = dot product (computationally cheaper).
- **Semantic alignment**: Text embedding models are trained to make semantically similar texts have high cosine similarity.

---

## 11. Prompt Engineering Architecture

### Single-Turn Prompt Structure

```
┌──────────────────────────────────────────────────────────────┐
│ ROLE: user                                                    │
│                                                              │
│ System Instruction:                                          │
│ "You are a helpful assistant. Use the following context       │
│  to answer the question. If the context doesn't contain      │
│  relevant information, say so clearly."                      │
│                                                              │
│ Context:                                                     │
│ [Retrieved chunk 1]                                          │
│ [Retrieved chunk 2]                                          │
│ [Retrieved chunk 3]                                          │
│                                                              │
│ Question: [User's question]                                  │
└──────────────────────────────────────────────────────────────┘
```

### Multi-Turn Prompt Structure

```
┌──────────────────────────────────────┐
│ ROLE: system                          │
│ "You are a helpful assistant.         │
│  Use the following retrieved context  │
│  to inform your answers..."           │
│                                       │
│ Context:                              │
│ [Retrieved chunks]                    │
├──────────────────────────────────────┤
│ ROLE: user                            │
│ "Tell me about Yash."                 │
├──────────────────────────────────────┤
│ ROLE: assistant                       │
│ "Yash is interested in cloud..."      │
├──────────────────────────────────────┤
│ ROLE: user                            │
│ "What technologies does he know?"     │
└──────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|---|---|
| Context in user message (single-turn) | Simpler prompt, no system message overhead for one-shot queries |
| Context in system message (multi-turn) | Context persists across turns without repeating; conversation history stays clean |
| Explicit "say so" instruction | Prevents hallucination when context is irrelevant |
| Top-k=3 default | Balances context richness vs. context window size |

### Context Window Considerations

```
qwen2.5:0.5b context window: ~32K tokens

Typical breakdown:
- System instruction: ~50 tokens
- Retrieved context (3 chunks × ~100 words): ~400 tokens
- User question: ~20 tokens
- Available for response: ~31,530 tokens

→ Context window is NOT a bottleneck for this use case
```

---

## 12. Middleware Pipeline

### Execution Order (Detailed)

```
                     REQUEST PATH                    RESPONSE PATH
                     ============                    =============

Browser/Client
    │                                                      ▲
    ▼                                                      │
┌──────────────────────────────────────────────────────────────┐
│ CORS MIDDLEWARE                                              │
│ • Checks Origin header against CORS_ORIGINS                  │
│ • On preflight (OPTIONS): returns 200 immediately            │
│ • On normal request: passes through, adds CORS headers to    │
│   response on the way back                                   │
└──────────┬───────────────────────────────────────▲───────────┘
           │                                       │
           ▼                                       │
┌──────────────────────────────────────────────────────────────┐
│ API KEY MIDDLEWARE                                            │
│ • If API_KEY_ENABLED=false → no-op (pass through)            │
│ • If path in {/health, /docs, /redoc, /openapi.json} → pass  │
│ • Check X-API-Key header:                                    │
│   • Missing/wrong → return 401 (SHORT CIRCUIT)               │
│   • Correct → pass through                                   │
└──────────┬───────────────────────────────────────▲───────────┘
           │                                       │
           ▼                                       │
┌──────────────────────────────────────────────────────────────┐
│ RATE LIMITER (per-route decorator)                           │
│ • Extract client IP from request                             │
│ • Check against configured limit (e.g., 60/minute)           │
│ • Over limit → return 429 (SHORT CIRCUIT)                    │
│ • Under limit → pass through                                │
└──────────┬───────────────────────────────────────▲───────────┘
           │                                       │
           ▼                                       │
┌──────────────────────────────────────────────────────────────┐
│ ROUTE HANDLER                                                │
│ • Pydantic validation → 422 if invalid                       │
│ • Business logic → service calls                             │
│ • Response construction                                      │
└──────────────────────────────────────────────────────────────┘
```

### Short Circuit Behavior

The middleware chain can short-circuit at any layer:
- **CORS**: OPTIONS preflight → 200 immediately (no auth/rate check).
- **Auth**: Bad/missing key → 401 (no rate limit consumption, no route handler execution).
- **Rate Limit**: Over limit → 429 (no route handler execution).

This is efficient — rejected requests are handled as early as possible with minimal computation.

---

## 13. Error Handling Architecture

### Error Response Taxonomy

| HTTP Code | Source | Meaning | Example |
|---|---|---|---|
| **400** | Route handler | Client error — bad input | Empty content, non-.txt file |
| **401** | Auth middleware | Authentication failure | Missing/wrong API key |
| **404** | Route handler | Resource not found | Delete non-existent user's docs |
| **422** | Pydantic | Validation failure | Missing required field, type mismatch |
| **429** | Rate limiter | Too many requests | Exceeded 60/minute on RAG endpoints |
| **502** | Service layer | Upstream failure | Ollama/ChromaDB unreachable |

### Error Propagation

```
Service Layer               Route Layer               Middleware Layer
   │                           │                           │
   ├── ChromaDB fails          │                           │
   │   └── HTTPException(502)──▶ Caught by FastAPI         │
   │                           │ → JSON {"detail": "..."}  │
   │                           │                           │
   ├── Ollama fails            │                           │
   │   └── HTTPException(502)──▶ Caught by FastAPI         │
   │                           │                           │
   │                           ├── Pydantic validation     │
   │                           │   └── 422 auto-generated  │
   │                           │                           │
   │                           │                           ├── Auth fails
   │                           │                           │   └── JSONResponse(401)
   │                           │                           │
   │                           │                           ├── Rate exceeded
   │                           │                           │   └── 429 + Retry-After
```

### Why HTTP 502 for Upstream Failures?

HTTP 502 (Bad Gateway) semantically means "the server, while acting as a gateway or proxy, received an invalid response from an upstream server." Since the RAG API acts as a gateway to both Ollama and ChromaDB, 502 is the correct status code when those services fail.

---

## 14. Configuration Architecture

### Configuration Flow

```
                    Priority (highest → lowest)
                    ═══════════════════════════

    ┌────────────────────┐
    │ Environment Vars   │  ← Production (Docker, K8s)
    │ OLLAMA_URL=...     │
    └────────┬───────────┘
             │ overrides
    ┌────────▼───────────┐
    │ .env File          │  ← Development (local)
    │ OLLAMA_URL=...     │
    └────────┬───────────┘
             │ overrides
    ┌────────▼───────────┐
    │ Default Values     │  ← Code (fallback)
    │ OLLAMA_URL=http... │
    └────────────────────┘
```

### 12-Factor App Compliance

| Factor | How We Comply |
|---|---|
| **III. Config** | All configuration in environment variables via `pydantic-settings` |
| **IV. Backing Services** | Ollama and ChromaDB are treated as attached resources, configurable via URL |
| **VI. Processes** | Stateless processes; state lives in ChromaDB (backing service) |
| **XI. Logs** | Logs emitted to stdout via Python's `logging` module |
| **XII. Admin Processes** | `build_knowledge_base.py` runs as a one-off admin process |

---

## 15. Testing Architecture

### Test Pyramid

```
         ╱╲
        ╱  ╲         Manual Testing
       ╱    ╲        (Swagger UI, curl)
      ╱──────╲
     ╱        ╲      Integration Tests
    ╱   test_  ╲     (test_rag.py, test_documents.py,
   ╱   health   ╲     test_streaming.py, test_auth.py)
  ╱──────────────╲
 ╱                ╲   Unit Tests
╱  test_chunking   ╲  (Pure function tests, no I/O)
╱                    ╲
──────────────────────
```

### Test Isolation Strategy

```
┌──────────────────────────────────────┐
│         Test Execution               │
│                                      │
│  1. conftest.py patches ollama.list  │
│     (prevents real network calls)    │
│                                      │
│  2. Test runs                        │
│     (may create test docs in DB)     │
│                                      │
│  3. autouse cleanup fixture runs     │
│     (deletes docs with user_name=    │
│      "test_user_pytest")             │
│                                      │
│  4. Next test starts clean           │
└──────────────────────────────────────┘
```

### Mocking Boundaries

```
┌────────────────────────────────────────────────────────┐
│                    NOT MOCKED                           │
│  (Real implementations used in tests)                   │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ FastAPI       │  │ Pydantic     │  │ ChromaDB     │ │
│  │ Router        │  │ Validation   │  │ (embedded)   │ │
│  │ Middleware     │  │              │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                        │
├────────────────────────────────────────────────────────┤
│                     MOCKED                              │
│  (Replaced with unittest.mock)                          │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ Ollama LLM   │  │ ollama.list  │                    │
│  │ (_async_     │  │ (startup     │                    │
│  │  client)     │  │  check)      │                    │
│  └──────────────┘  └──────────────┘                    │
└────────────────────────────────────────────────────────┘
```

---

## 16. Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────┐
│              Developer Machine           │
│                                         │
│  ┌───────────────┐   ┌───────────────┐  │
│  │  Terminal 1    │   │  Terminal 2    │  │
│  │               │   │               │  │
│  │  Ollama       │   │  uvicorn      │  │
│  │  serve        │   │  --reload     │  │
│  │  :11434       │   │  :8000        │  │
│  └───────────────┘   └───────┬───────┘  │
│                              │          │
│                    ┌─────────▼────────┐ │
│                    │   ./chroma_db/   │ │
│                    │   (local disk)   │ │
│                    └──────────────────┘ │
└─────────────────────────────────────────┘
```

### Docker Deployment

```
┌─────────────────────────────────────────┐
│               Host Machine               │
│                                         │
│  ┌───────────────┐                      │
│  │  Ollama       │                      │
│  │  (native)     │                      │
│  │  :11434       │                      │
│  └───────┬───────┘                      │
│          │ host.docker.internal:11434    │
│  ┌───────▼───────────────────────────┐  │
│  │         Docker Container           │  │
│  │                                   │  │
│  │  ┌─────────────┐                  │  │
│  │  │  RAG API    │                  │  │
│  │  │  uvicorn    │                  │  │
│  │  │  :8000      │─── -p 8000:8000  │  │
│  │  └──────┬──────┘                  │  │
│  │         │                         │  │
│  │  ┌──────▼──────┐                  │  │
│  │  │ /app/       │                  │  │
│  │  │ chroma_db/  │── VOLUME mount   │  │
│  │  └─────────────┘                  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Production Deployment (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────┐     ┌──────────────┐     ┌────────────┐  │
│  │  Nginx   │────▶│  RAG API     │────▶│  Ollama    │  │
│  │  (TLS    │     │  (Gunicorn + │     │  (GPU      │  │
│  │   proxy) │     │   Uvicorn    │     │   server)  │  │
│  │  :443    │     │   workers)   │     │  :11434    │  │
│  └──────────┘     │  :8000       │     └────────────┘  │
│                   └──────┬───────┘                      │
│                          │                              │
│                   ┌──────▼───────┐                      │
│                   │   ChromaDB   │                      │
│                   │   (volume)   │                      │
│                   └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

---

## 17. Security Architecture

### Security Layers

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Network Security                                │
│ • CORS restricts browser origins                         │
│ • TLS encryption (via reverse proxy in production)       │
│ • Docker network isolation                               │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Authentication                                  │
│ • Optional API key in X-API-Key header                   │
│ • Public paths whitelisted (/health, /docs)              │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Rate Limiting                                   │
│ • Per-IP rate limits prevent abuse                        │
│ • Lower limits on expensive RAG endpoints                │
├─────────────────────────────────────────────────────────┤
│ Layer 4: Input Validation                                │
│ • Pydantic enforces type safety and constraints           │
│ • min_length, ge/le validators on all inputs              │
│ • Regex pattern on ChatMessage.role                       │
├─────────────────────────────────────────────────────────┤
│ Layer 5: Error Handling                                   │
│ • No stack traces in production responses                 │
│ • Structured error messages ({"detail": "..."})           │
│ • Upstream failures → 502, never raw exceptions           │
└─────────────────────────────────────────────────────────┘
```

### Threat Model

| Threat | Mitigation | Status |
|---|---|---|
| Unauthorized access | API key middleware | ✅ Implemented |
| DDoS / abuse | Rate limiting per IP | ✅ Implemented |
| Cross-origin attacks | CORS configuration | ✅ Implemented |
| Data injection | Pydantic validation | ✅ Implemented |
| Prompt injection | LLM instruction prefix | ⚠️ Basic |
| Timing attacks on auth | `hmac.compare_digest` | ❌ Not yet |
| Data leakage across users | Metadata filtering | ✅ Implemented |

---

## 18. Scalability & Evolution Roadmap

### Current Bottlenecks

```
                    Latency Budget for POST /ask
                    ════════════════════════════

  ┌──────────────────────────────────────────────────┐
  │ Pydantic validation         │  < 1ms             │
  │ Rate limit check            │  < 1ms             │
  │ Question embedding          │  ~50ms    ████     │
  │ ChromaDB similarity search  │  ~10ms    ██       │
  │ LLM inference (qwen2.5:0.5b)│  ~500ms  █████████ │ ← Bottleneck
  │ Response serialization      │  < 1ms             │
  └──────────────────────────────────────────────────┘
  Total: ~560ms per request
```

### Scaling Strategies

| Strategy | Approach | Effort |
|---|---|---|
| **GPU Acceleration** | Run Ollama with CUDA/Metal GPU support | Low — config change only |
| **Larger Models** | Swap `qwen2.5:0.5b` for `llama3:8b` or `mistral:7b` | Low — env var change |
| **Response Caching** | Redis cache for repeated question+context pairs | Medium |
| **Horizontal Scaling** | Multiple API instances + shared ChromaDB (HttpClient) | Medium |
| **Distributed Vector DB** | Replace ChromaDB with Qdrant, Pinecone, or Weaviate | High |
| **Batch Embedding** | Embed multiple queries in one Ollama call | Medium |
| **True Semantic Chunking** | Embedding-based similarity for chunk boundaries | Medium |

### Future Architecture (Distributed)

```
                     ┌─────────────┐
                     │  Load       │
                     │  Balancer   │
                     └──────┬──────┘
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │  API #1  │ │  API #2  │ │  API #3  │
         └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
         ┌────▼────────────▼────────────▼────┐
         │           Redis Cache              │
         └────────────────┬───────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼                       ▼
        ┌──────────┐           ┌──────────┐
        │  Qdrant  │           │  Ollama  │
        │  Cluster │           │  Cluster │
        └──────────┘           └──────────┘
```

---

## 19. Architecture Decision Records (ADRs)

### ADR-001: Local LLM Over Cloud APIs

**Context**: Need an LLM for answer generation.

**Decision**: Use Ollama (local) instead of OpenAI/Anthropic APIs.

**Rationale**:
- No API keys or costs.
- Full data privacy — no personal data leaves the machine.
- Works offline.
- Predictable latency (no network variability).

**Trade-offs**:
- Limited to models that fit in local memory.
- Lower quality than GPT-4 (but qwen2.5:0.5b is sufficient for personal Q&A).

---

### ADR-002: Embedded ChromaDB Over External Vector DB

**Context**: Need a vector database for document storage and similarity search.

**Decision**: Use ChromaDB `PersistentClient` (embedded) over a remote vector database.

**Rationale**:
- Zero infrastructure overhead — no separate server to deploy.
- Persistent to disk — survives restarts.
- Fast for small-to-medium datasets (<100K documents).
- Simple Python API.

**Trade-offs**:
- Single-process access only (no concurrent writes from multiple API instances).
- No built-in replication or horizontal scaling.

---

### ADR-003: `asyncio.to_thread()` for Blocking Calls

**Context**: ChromaDB and some Ollama operations are synchronous (blocking).

**Decision**: Wrap blocking calls in `asyncio.to_thread()` instead of running them directly in async handlers.

**Rationale**:
- Prevents event loop starvation.
- Allows concurrent request handling despite blocking I/O.
- Standard Python pattern for integrating sync libraries into async frameworks.

**Trade-offs**:
- Thread pool overhead (~1ms per call).
- Default thread pool size (OS-dependent, typically ~32 threads) limits concurrency.

---

### ADR-004: SSE Over WebSockets for Streaming

**Context**: Need real-time token streaming from LLM to client.

**Decision**: Use Server-Sent Events (SSE) instead of WebSockets.

**Rationale**:
- Unidirectional (server → client) matches the use case.
- Works over standard HTTP — no upgrade handshake.
- Browser-native auto-reconnect.
- Simpler server implementation.

**Trade-offs**:
- No bidirectional communication (not needed).
- Maximum ~6 concurrent SSE connections per browser per domain (browser limit).

---

### ADR-005: Deterministic Chunk IDs

**Context**: Need a strategy for document chunk IDs in ChromaDB.

**Decision**: Use `{user_name}-chunk{i}` as deterministic IDs.

**Rationale**:
- Re-ingestion replaces old chunks instead of creating duplicates (upsert semantics).
- Predictable IDs simplify debugging and testing.
- User-scoped → prevents cross-user ID collisions.

**Trade-offs**:
- If user re-ingests with fewer chunks, old higher-indexed chunks persist.
- If user re-ingests with a different chunking strategy, chunk boundaries shift.

---

## 20. Interview Deep-Dive Questions

### System Design

**Q: Walk me through what happens when a user asks "What are Yash's skills?"**
> See [Section 6: Request Lifecycle](#6-request-lifecycle) for the 12-step detailed flow.

**Q: How would you add a caching layer to this system?**
> I'd add Redis as a response cache. The cache key would be a hash of `(question, user, n_results)`. Before calling retrieval + LLM, check the cache. On cache hit, return immediately. On cache miss, execute the full pipeline and cache the result with a TTL (e.g., 5 minutes). For streaming endpoints, I'd cache the complete response and re-stream from cache.

**Q: How would you add multi-user authentication with JWT tokens?**
> Replace the simple API key middleware with JWT-based auth: (1) Add a `/auth/login` endpoint that validates credentials and returns a JWT. (2) The middleware extracts the JWT from the `Authorization: Bearer <token>` header. (3) Decode the JWT to get the `user_id`. (4) Inject `user_id` into the request state. (5) Routes use `request.state.user_id` instead of requiring `user` in the body. This scopes all operations to the authenticated user automatically.

**Q: What would break if you had 1 million documents?**
> (1) `GET /admin/stats` fetches ALL documents — would need pagination or aggregation queries. (2) ChromaDB's embedded mode would struggle — migrate to Qdrant or ChromaDB's server mode. (3) Embedding generation during ingestion would be slow — need batch processing. (4) HNSW index memory consumption grows linearly — may need quantization or disk-based indices.

### Code Architecture

**Q: Why did you split `main.py` into multiple modules?**
> The v1 `main.py` was 380 lines and growing. As features were added (streaming, auth, rate limiting, chunking strategies), the monolith became hard to navigate, test, and maintain. The v2 split follows the Single Responsibility Principle: each module handles one concern. This enables independent testing (e.g., chunking strategies are pure functions with no I/O dependency), parallel development, and easier debugging.

**Q: Why is `services/chunking.py` a pure module with no imports from the project?**
> By design. Chunking is a **pure computation** — it takes text and produces chunks with no side effects, no database calls, no network I/O. This makes it: (1) trivially testable (no mocking needed), (2) reusable outside the API (the CLI tool imports it), (3) fast (no async overhead), and (4) easy to reason about.

**Q: How does `__getattr__` in `database.py` work and why is it there?**
> Python calls `__getattr__` on a module when a normal attribute lookup fails. When code does `from database import collection`, Python looks for `collection` as a module attribute. Since it's not defined at the top level, `__getattr__` is called, which returns `get_collection()`. This provides backward compatibility with v1 code that imported `collection` directly, while the new preferred API is `get_collection()`. It's also lazy — the collection isn't created until first access.
