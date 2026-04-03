<!-- Auto-generated guidance for AI coding agents. Update as needed. -->
# Copilot / AI agent instructions for this repo

A production-style RAG Q&A system built with LangChain, FAISS, Google Gemini, and FastAPI.
Supports two modes: general document Q&A and bioinformatics literature Q&A.

## 1. Big picture

- **Purpose**: End-to-end RAG pipeline — ingest PDFs/DOCX, embed into FAISS, retrieve with MMR, answer with Gemini.
- **Runtime**: Python 3.11. Environment variables via `python-dotenv` (`.env` file).
- **Domains**: `general` (800-token chunks) and `bioinfo` (500-token chunks, domain-constrained prompt).

## 2. Key files

| File | Role |
|---|---|
| `app.py` | FastAPI app — `/status`, `/upload`, `/ask` endpoints |
| `src/ingestion.py` | PDF/DOCX loader + chunker (domain-aware chunk sizes) |
| `src/vectorstore.py` | FAISS index management with Google Generative AI embeddings |
| `src/retriever.py` | MMR retrieval wrapper (lambda_mult=0.7) |
| `src/rag_chain.py` | Gemini LLM + prompt templates (general + bioinfo) |
| `src/day1test.py` | Original prototype (ChromaDB + HuggingFace, kept for reference) |

## 3. How to run

```bash
# Install deps
pip install -r requirements.txt

# Start API server
uvicorn app:app --reload

# Open interactive docs
open http://localhost:8000/docs
```

## 4. Environment

`.env` must contain:
```
GOOGLE_API_KEY=your_key_here
```

## 5. Embedding model

Uses `models/text-embedding-004` via `GoogleGenerativeAIEmbeddings` (no PyTorch required).
Previously used `all-MiniLM-L6-v2` (HuggingFace) — switched due to PyTorch version incompatibility.

## 6. Docker

```bash
docker-compose up --build
```

Volumes persist `data/` (uploaded files) and `vectorstore/` (FAISS index) across restarts.

## 7. Conventions

- Add new domain modes by: (a) adding a chunk size branch in `ingestion.py`, (b) adding a prompt in `rag_chain.py::PROMPTS`.
- Never commit `.env`, `vectorstore/`, `data/`, or `__pycache__/`.
- Keep `src/day1test.py` as a standalone runnable prototype — do not import from it.
