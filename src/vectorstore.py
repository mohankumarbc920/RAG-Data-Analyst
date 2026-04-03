"""
Vector Store Module
Manages FAISS index with Google Generative AI embeddings.
Supports incremental document addition and persistent local storage.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Google's text embedding model — no PyTorch required, uses GOOGLE_API_KEY
EMBED_MODEL = "models/text-embedding-004"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def load_store(index_dir: str) -> FAISS | None:
    """Load an existing FAISS index from disk. Returns None if not found."""
    index_file = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(
            index_dir,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return None


def add_documents(docs: list, index_dir: str) -> int:
    """
    Embed documents and add to FAISS index.
    Creates a new index if one doesn't exist; merges into existing otherwise.
    Returns the total number of vectors now in the index.
    """
    os.makedirs(index_dir, exist_ok=True)
    embeddings = get_embeddings()

    existing = load_store(index_dir)
    if existing:
        existing.add_documents(docs)
        existing.save_local(index_dir)
        return existing.index.ntotal
    else:
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(index_dir)
        return store.index.ntotal
