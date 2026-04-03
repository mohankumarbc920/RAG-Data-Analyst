# RAG Q&A Assistant

![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![LangChain](https://img.shields.io/badge/LangChain-0.2-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-red)
![Python](https://img.shields.io/badge/Python-3.11-yellow)
![Docker](https://img.shields.io/badge/Docker-Containerized-lightblue)

A document question-answering system that lets you upload PDFs and ask questions about them. It only answers from your documents — no made-up facts.

Supports two modes:
- **General** — any PDF or DOCX document
- **Bioinfo** — bioinformatics research papers (specialized prompts)

---

## How it works

```
Upload PDF → Split into chunks → Store as vectors (FAISS)
                                        ↓
Ask question → Find relevant chunks → Send to Gemini → Get answer with citations
```

---

## Tech Stack

| What | Tool |
|---|---|
| API server | FastAPI |
| PDF/DOCX loading | LangChain |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | Google Gemini Embedding |
| Vector search | FAISS |
| LLM | Google Gemini 2.5 Flash |
| Deployment | Docker |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/mohankumarbc920/RAG-Data-Analyst.git
cd RAG-Data-Analyst
```

### 2. Create a `.env` file

```bash
cp .env.example .env
```

Then open `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at: https://aistudio.google.com

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
uvicorn app:app --reload
```

Open your browser at: **http://localhost:8000/docs**

---

## API Endpoints

### `POST /upload` — Upload a document
- Upload a PDF or DOCX file
- Set `domain` to `general` or `bioinfo`

### `POST /ask` — Ask a question
- Type your question
- Set the same `domain` you used when uploading
- Returns the answer + which pages it came from

### `GET /status` — Check how many documents are loaded

---

## Example

Upload a biology paper, then ask:

```
query:  What genes were studied in this paper?
domain: bioinfo
```

Response:
```json
{
  "answer": "The study focused on BRCA1 and BRCA2 genes... [HiCGNN.pdf | Page 4]",
  "sources": ["HiCGNN.pdf"],
  "chunks_used": 5,
  "latency_seconds": 1.3
}
```

---

## Run with Docker

```bash
docker-compose up --build
```

This keeps your uploaded files and vector index saved even if you restart.

---

## Project Structure

```
app.py                  → API server (upload, ask, status)
src/
  ingestion.py          → Load and split documents
  vectorstore.py        → Store and search document vectors
  retriever.py          → Find relevant chunks for a query
  rag_chain.py          → Build prompt and call Gemini
Dockerfile
docker-compose.yml
requirements.txt
```
