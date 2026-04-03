"""
Document Ingestion Pipeline
Loads PDF/DOCX files, cleans text, and splits into chunks with metadata.
"""

import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    """Load PDF or DOCX file into LangChain Documents."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use PDF or DOCX.")
    return loader.load()


def clean_text(text: str) -> str:
    """Remove excessive whitespace, repeated newlines, and junk characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # strip non-ASCII
    return text.strip()


def load_and_chunk(file_path: str, domain: str = "general") -> list:
    """
    Load a document and split into overlapping chunks.

    Chunk settings:
      - general: 800 tokens, 200 overlap
      - bioinfo: 500 tokens, 100 overlap (tighter — bio docs are dense)

    Returns a list of LangChain Document objects with metadata:
      source, page, domain
    """
    docs = load_document(file_path)

    # Domain-specific chunking parameters
    if domain == "bioinfo":
        chunk_size = 500
        chunk_overlap = 100
    else:
        chunk_size = 800
        chunk_overlap = 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Clean and enrich metadata
    enriched = []
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
        if len(chunk.page_content) < 50:  # skip near-empty chunks
            continue
        chunk.metadata.setdefault("source", file_path)
        chunk.metadata["domain"] = domain
        enriched.append(chunk)

    return enriched
