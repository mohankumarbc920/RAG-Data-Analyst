"""
RAG Q&A Assistant — FastAPI Backend

Endpoints:
  POST /upload  — Ingest a PDF or DOCX into the FAISS vector store
  POST /ask     — Ask a question against indexed documents
  GET  /status  — Check how many vectors are indexed
"""

import os
import shutil
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.ingestion import load_and_chunk
from src.vectorstore import add_documents, load_store
from src.retriever import get_retriever
from src.rag_chain import ask

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Q&A Assistant",
    description="Domain-aware RAG system — supports general documents and bioinformatics literature.",
    version="1.0.0",
)

UPLOAD_DIR = "data"
FAISS_DIR = "vectorstore"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Upload a PDF or DOCX.",
        )
    return ext


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    """Returns the number of vectors currently in the FAISS index."""
    store = load_store(FAISS_DIR)
    vector_count = store.index.ntotal if store else 0
    return {
        "status": "ready" if vector_count > 0 else "empty",
        "vectors_indexed": vector_count,
        "index_dir": FAISS_DIR,
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form("general", description="'general' or 'bioinfo'"),
):
    """
    Ingest a document into the vector store.

    - **file**: PDF or DOCX file to upload
    - **domain**: Use 'bioinfo' for bioinformatics papers; defaults to 'general'
    """
    _validate_extension(file.filename)

    # Persist file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    t0 = time.perf_counter()
    try:
        chunks = load_and_chunk(save_path, domain=domain)
        total_vectors = add_documents(chunks, FAISS_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    elapsed = round(time.perf_counter() - t0, 2)

    return {
        "message": f"Successfully ingested '{file.filename}'",
        "chunks_created": len(chunks),
        "total_vectors_in_index": total_vectors,
        "domain": domain,
        "elapsed_seconds": elapsed,
    }


@app.post("/ask")
async def ask_question(
    query: str = Form(..., description="Your question"),
    domain: str = Form("general", description="'general' or 'bioinfo'"),
    top_k: int = Form(5, ge=1, le=20, description="Number of chunks to retrieve"),
):
    """
    Ask a question against indexed documents.

    - **query**: The question to answer
    - **domain**: Match to the domain used during upload for best results
    - **top_k**: How many chunks to retrieve (default 5)
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.perf_counter()
    retriever = get_retriever(FAISS_DIR, top_k=top_k)
    result = ask(retriever, query, domain=domain)
    result["latency_seconds"] = round(time.perf_counter() - t0, 3)

    return JSONResponse(content=result)
