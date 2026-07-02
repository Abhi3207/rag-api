# RAG-API Comprehensive Improvement Plan

## Current State

The project is a minimal RAG (Retrieval-Augmented Generation) API with:
- **main.py** — FastAPI app with 2 endpoints (`POST /documents`, `GET /ask`)
- **build_knowledge_base.py** — Script to load `profile.txt` into ChromaDB
- **profile.txt** — Sample profile data
- Hardcoded config (model names, DB paths, Ollama URLs)
- No error handling, no tests, no requirements file, barebones README



## Proposed Changes, need to be implemented

The goal is to transform this into a **production-grade, well-structured RAG API** with expanded scope and clean architecture.

---

### 1. Project Structure & Configuration

#### [NEW] [config.py](file:///c:/Users/abhin/Documents/rag-api/config.py)
- Centralized configuration using Pydantic `BaseSettings` with env variable support
- Configurable: `OLLAMA_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`, `CHROMA_DB_PATH`, `COLLECTION_NAME`, `DEFAULT_N_RESULTS`

#### [NEW] [requirements.txt](file:///c:/Users/abhin/Documents/rag-api/requirements.txt)
- Pin all dependencies: `fastapi`, `uvicorn`, `ollama`, `chromadb`, `pydantic-settings`, `python-multipart`

#### [NEW] [.env.example](file:///c:/Users/abhin/Documents/rag-api/.env.example)
- Template for environment variables

---

### 2. Core API Improvements

#### [MODIFY] [main.py](file:///c:/Users/abhin/Documents/rag-api/main.py)
Major enhancements:
- **Import config** from `config.py` instead of hardcoding values
- **`GET /health`** — Health check endpoint (checks Ollama & ChromaDB connectivity)
- **`POST /documents/upload`** — Upload a `.txt` file directly instead of raw JSON
- **`GET /documents`** — List all stored documents/chunks with metadata
- **`DELETE /documents/{user_name}`** — Delete all chunks for a specific user
- **`POST /ask`** — Change from GET to POST, add `n_results` parameter, add system prompt, return source metadata
- **`POST /chat`** — Conversational endpoint that accepts message history for multi-turn conversations
- **Proper error handling** — Try/except blocks with meaningful HTTPException errors
- **CORS middleware** — Enable cross-origin requests for frontend integration
- **Structured response models** — Pydantic models for all responses
- **Startup event** — Validate Ollama connection on boot

#### [MODIFY] [build_knowledge_base.py](file:///c:/Users/abhin/Documents/rag-api/build_knowledge_base.py)
- Use config from `config.py`
- Add `argparse` for specifying input file and user name from CLI
- Add duplicate detection (skip if chunks already exist)
- Better chunking with configurable chunk size

---

### 3. Testing & Quality

#### [NEW] [tests/test_api.py](file:///c:/Users/abhin/Documents/rag-api/tests/test_api.py)
- Unit tests using `pytest` + FastAPI `TestClient`
- Test health check, document CRUD, ask endpoint, error cases

#### [NEW] [tests/__init__.py](file:///c:/Users/abhin/Documents/rag-api/tests/__init__.py)
- Package init

---

### 4. Documentation & DevOps

#### [MODIFY] [README.md](file:///c:/Users/abhin/Documents/rag-api/README.md)
- Project overview with architecture diagram
- Prerequisites (Ollama, Python 3.10+)
- Setup & installation instructions
- API endpoint documentation with request/response examples
- Environment variables reference
- How to run tests

#### [NEW] [Dockerfile](file:///c:/Users/abhin/Documents/rag-api/Dockerfile)
- Multi-stage Docker build for containerized deployment

#### [MODIFY] [.gitignore](file:///c:/Users/abhin/Documents/rag-api/.gitignore)
- Add `chroma_db/`, `.env`, `*.log` patterns

---

### 5. Git Push

After all changes:
```bash
git add -A
git commit -m "feat: major upgrade - config management, extended API, tests, Docker, docs"
git push origin main
```

## Verification Plan

### Manual Verification
- Confirm all files are created and well-structured
- Verify git commit is clean and push succeeds

> [!IMPORTANT]
> This plan assumes Ollama is available locally at `http://localhost:11434`. The tests will mock the Ollama calls so they run without a live instance.

## Open Questions

> [!NOTE]
> **Profile data**: The current `profile.txt` has placeholder-style commas (e.g., "I have experience with , Python, JavaScript"). Should I clean this up or leave it as-is since it's sample data?
