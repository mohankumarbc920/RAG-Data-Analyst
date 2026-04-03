from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# --- Step 1: Fake a small "document" ---
raw_text = """
Apple Inc reported revenue of $94.9 billion in Q3 2024, a 5% increase year over year.
iPhone sales accounted for 46% of total revenue at $43.8 billion.
Services revenue hit a record $24.2 billion, growing 14% year over year.
Mac revenue was $7 billion, while iPad revenue came in at $7.2 billion.
CEO Tim Cook highlighted strong growth in emerging markets, especially India.
"""

# --- Step 2: Chunk it ---
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.create_documents([raw_text])
print(f"Created {len(chunks)} chunks")

# --- Step 3: Embed using free local model ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
print("Stored in ChromaDB!")

# --- Step 4: Ask a question ---
question = "What was the iPhone revenue?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
relevant_chunks = retriever.invoke(question)

print(f"\nTop relevant chunks for: '{question}'")
for i, chunk in enumerate(relevant_chunks):
    print(f"\nChunk {i+1}: {chunk.page_content}")

# --- Step 5: Send to Gemini LLM with context ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

context = "\n".join([c.page_content for c in relevant_chunks])
prompt = f"""Answer the question based only on the context below.
Context: {context}
Question: {question}
Answer:"""

response = llm.invoke(prompt)
print(f"\n Final Answer: {response.content}")