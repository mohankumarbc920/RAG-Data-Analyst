"""
Vector Store Module
Manages FAISS index with Google Generative AI embeddings.
Processes documents in batches to stay within free-tier rate limits (100 req/min).
"""

import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBED_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 80   # stay under the 100 req/min free-tier limit
BATCH_SLEEP = 65  # seconds to wait between batches


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
    Embed documents in batches and save to FAISS index.
    Sleeps between batches to respect the free-tier 100 req/min rate limit.
    Returns total number of vectors in the index.
    """
    os.makedirs(index_dir, exist_ok=True)
    embeddings = get_embeddings()

    store = load_store(index_dir)

    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        if idx > 0:
            print(f"Rate limit pause: waiting {BATCH_SLEEP}s before batch {idx + 1}/{len(batches)}...")
            time.sleep(BATCH_SLEEP)

        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)

    if store:
        store.save_local(index_dir)
        return store.index.ntotal
    return 0
