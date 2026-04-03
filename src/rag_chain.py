"""
RAG Chain — LLM + Prompt Templates + Answer Generation

Two modes:
  - general:  Generic document Q&A with source citations
  - bioinfo:  Domain-constrained bioinformatics assistant

Both prompts enforce grounding: answer ONLY from retrieved context.
Structured output includes answer, sources, and chunk count.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

GENERAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful Q&A assistant.
Answer ONLY using the context below. Do not use prior knowledge.
If the answer is not in the context, say "I don't know based on the provided documents."
Always cite the source document and page number when referencing information.

Context:
{context}

Question: {question}

Answer (with citations):""",
)

BIOINFO_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a bioinformatics expert assistant.
Answer ONLY using scientific evidence from the context below.
If the answer is not present in the context, say "Not found in provided literature."
Cite the source paper/document and page number for every claim.

Context:
{context}

Question: {question}

Scientific Answer (with citations):""",
)

PROMPTS = {
    "general": GENERAL_PROMPT,
    "bioinfo": BIOINFO_PROMPT,
}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,            # deterministic — reduces hallucinations
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ---------------------------------------------------------------------------
# RAG Answer Function
# ---------------------------------------------------------------------------

def ask(retriever, query: str, domain: str = "general") -> dict:
    """
    Retrieve relevant chunks, build grounded prompt, call Gemini, return result.

    Returns:
        {
          "answer":       str,
          "sources":      list[str],
          "chunks_used":  int,
          "domain":       str,
        }
    """
    if retriever is None:
        return {
            "answer": "No documents have been indexed yet. Please upload documents via /upload first.",
            "sources": [],
            "chunks_used": 0,
            "domain": domain,
        }

    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "No relevant content found for your query in the indexed documents.",
            "sources": [],
            "chunks_used": 0,
            "domain": domain,
        }

    # Build context string with inline source labels
    context_parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[{source} | Page {page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Select prompt by domain (default to general for unknown domains)
    prompt_template = PROMPTS.get(domain, GENERAL_PROMPT)
    prompt = prompt_template.format(context=context, question=query)

    llm = get_llm()
    response = llm.invoke(prompt)

    # Deduplicated source list
    sources = list({
        os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs
    })

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(docs),
        "domain": domain,
    }
