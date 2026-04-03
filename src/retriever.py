"""
Retrieval Pipeline
Wraps FAISS with MMR (Max Marginal Relevance) for diverse, relevant results.

MMR balances:
  - Relevance: how similar chunks are to the query
  - Diversity: avoids returning near-duplicate chunks

lambda_mult closer to 1 = more relevance focus
lambda_mult closer to 0 = more diversity focus
"""

from langchain_core.vectorstores import VectorStoreRetriever
from src.vectorstore import load_store


def get_retriever(
    index_dir: str,
    top_k: int = 5,
    use_mmr: bool = True,
    lambda_mult: float = 0.7,
) -> VectorStoreRetriever | None:
    """
    Build a retriever from the FAISS index.

    Args:
        index_dir:   Path to saved FAISS index.
        top_k:       Number of chunks to return.
        use_mmr:     Use Max Marginal Relevance (recommended).
        lambda_mult: MMR diversity tuning (0=diverse, 1=relevant-only).

    Returns None if no index has been built yet.
    """
    store = load_store(index_dir)
    if store is None:
        return None

    if use_mmr:
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": top_k * 3,   # fetch wider pool before MMR re-ranks
                "lambda_mult": lambda_mult,
            },
        )

    return store.as_retriever(search_kwargs={"k": top_k})
