# Multilingual Indic RAG Pipeline with Automated Evaluations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sourabhsonker-multilingual-rag.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

A production-grade, highly performant, and fully asynchronous Retrieval-Augmented Generation (RAG) system. This pipeline accepts user queries in **Hindi or Hinglish (Hindi-English code-mixed)**, performs semantic cross-lingual retrieval over an **English-only document corpus**, and generates highly grounded, faithful answers natively in Hindi or Hinglish.

🔗 **Live Interactive Application:** [sourabhsonker-multilingual-rag.streamlit.app](https://sourabhsonker-multilingual-rag.streamlit.app/)

---

---

##  Key Features

- **Asynchronous Ingestion Pipeline** — Concurrent file reading, Unicode normalization (NFC composition) to prevent character representation failures, and recursive chunking with configurable overlap.
- **Cross-Lingual Alignment** — Translates Hindi/Hinglish query semantics directly to English document vectors in a shared 384-dimensional aligned vector space, completely bypassing slow, lossy translation APIs.
- **Resilient Multi-Provider Generation** — Non-blocking, rate-limit-resilient REST client supporting both **Google Gemini 2.5 Flash** and **Sarvam AI** (e.g., `sarvam-30b`) models, with automatic graceful degradation.
- **Modern Vector Database** — Integrates a local **Qdrant** database utilizing its state-of-the-art **Unified Query API** (`query_points`) and deterministic UUID v5 point IDs to prevent duplication during re-ingestion.
- **Automated Ragas Evaluation Harness** — Programmatic pipeline that evaluates live runs against a static, human-curated **Golden Dataset**, scoring *Faithfulness* (hallucination detection) and *Context Recall* using a high-reasoning LLM judge (Llama 3.3 70B on Groq).
- **Production Servicing & Tooling**
  - Fully asynchronous REST API served via **FastAPI**, with strict Pydantic v2 schemas and global exception handling.
  - Multi-stage, ultra-optimized **Docker & Docker Compose** builds, reducing image size from 2 GB down to **250 MB** (using Debian-slim and copying pre-compiled virtual environments).
  - Modern packaging using **`uv`** for near-instant dependency locking and resolution.
  - Comprehensive testing with **`pytest` & `pytest-asyncio`**, including tests run against active Docker containers.

---

##  Tech Stack

| Category | Technology |
|---|---|
| Core Language | Python 3.12 + Asyncio |
| Model Server | FastAPI + Uvicorn + Pydantic v2 |
| Vector Database | Qdrant (Rust-based) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, 50+ languages) |
| Generator Models | Google Gemini 2.5 Flash / Sarvam AI / Groq (Llama 3.3 70B for evaluation) |
| Evaluation | Ragas Framework (Automated LLM-as-a-Judge) |
| Code Quality | Ruff (Linter) + Black (Formatter) + mypy (Type Checker) + pre-commit (Git Hooks) |
| Orchestration | Docker + Docker Compose (Multi-stage builds) |
| CI/CD | GitHub Actions (Automated Lint & Test workflows) |

---

##  System Architecture

```text
[ INGESTION FLOW ]
Raw Documents (.pdf) ──► PDFLoader ──► Unicode Normalizer ──► Recursive Chunker ──► Vector Generator ──► Qdrant DB

[ RUNTIME QUERY & GENERATION FLOW ]
Hinglish User Query
       │
       ▼ (clean_and_normalize_text)
Normalized Query Text
       │
       ▼ (EmbeddingModel.encode)
Query Vector (384D)
       │
       ▼ (Qdrant query_points)
Top-K Relevant English Chunks + Metadata
       │
       ▼ (Prompt Builder)
Context-Injected Multilingual Prompt
       │
       ▼ (httpx.AsyncClient → Gemini/Sarvam API)
Grounded Response (Hinglish/Hindi)
```

---

##  Quick Start (Launch in 3 Minutes)

### 1. Clone and configure environment

```bash
git clone https://github.com/yourusername/multilingual-rag.git
cd multilingual-rag
cp .env.example .env
```

Open your `.env` file and add your active API keys:

```env
GENERATOR_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here   # Required for running Ragas evaluations
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 2. Boot the containerized system

Build and launch your API server and Qdrant database concurrently:

```bash
docker compose up --build
```

This will compile your custom FastAPI image and boot Qdrant. Your server is live at `http://localhost:8000`.

### 3. Ingest a directory of PDFs

In a second terminal window, place any PDF manuals inside `data/raw/` and trigger the async ingestion:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "data/raw"}'
```

### 4. Query the API

Submit a Hinglish or Hindi query to get a grounded, cross-lingual answer with automatic source attribution:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Casual leave carry forward karne ke rules kya hain?", "top_k": 2}'
```

---

##  Testing Suite

We maintain a rigorous automated testing suite spanning isolated unit tests and Docker-backed integration tests. Ensure your Qdrant container is active, then run:

```bash
uv run pytest
```

**Passing tests output:**

```text
tests/integration/test_api.py ..                                                      [ 25%]
tests/integration/test_retrieval.py .                                                 [ 37%]
tests/integration/test_vector_store.py ..                                             [ 62%]
tests/unit/test_chunker.py ...                                                        [100%]

============================= 8 passed, 5 warnings in 128.68s =============================
```

---

##  Evaluation Scorecard (Ragas Results)

Our pipeline includes a resilient evaluation harness (`src/rag_pipeline/evaluation/run_eval.py`) that scores generated answers against a static, golden human dataset:

| Metric | Measured Score | Standard Target | Status |
|---|---|---|---|
| Faithfulness (Groundedness) | 0.9250 | > 0.90 | ✅ Excellent (no hallucinations) |
| Context Recall (Search Quality) | 1.0000 | > 0.90 | ✅ Perfect (found all facts) |

---

##  Directory Structure

```text
multilingual-rag/
├── .github/workflows/          # CI/CD pipelines (Lint and Test)
├── docker/
│   └── Dockerfile              # Multi-stage optimized build
├── src/
│   └── rag_pipeline/
│       ├── api/                # FastAPI routers and schemas
│       ├── embeddings/         # Local embedding model and batching pipeline
│       ├── evaluation/         # Ragas golden dataset and evaluation harness
│       ├── generation/         # Multilingual prompts and HTTP model client
│       ├── ingestion/          # Pathlib loading, Unicode preprocessing, chunking
│       └── retrieval/          # Qdrant client store wrapper and query search
├── tests/
│   ├── integration/            # Vector store, retrieval, and API endpoint integration tests
│   └── unit/                   # Isolated chunker and logic tests
├── data/
│   ├── raw/                    # Raw input PDF documents
│   └── eval/                   # Golden benchmark datasets and timestamped reports
├── pyproject.toml              # Project dependency configurations
└── docker-compose.yml          # Container orchestration configuration
```
