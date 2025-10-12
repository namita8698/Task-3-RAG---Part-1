# basic_rag_retrieval.py
# Local RAG using FAISS + HuggingFace embeddings (explicit allow_dangerous_deserialization)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# load .env (optional)
load_dotenv()

# Document loaders + splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local embeddings (HuggingFace wrapper)
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS vectorstore (community)
from langchain_community.vectorstores import FAISS

# Config
DOCS_DIR = Path("documents")
INDEX_DIR = Path("saved_index_local")  # using local index folder
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_index():
    print("🔹 Loading PDFs (local)...")
    docs = []
    if not DOCS_DIR.exists():
        print(f"[ERROR] documents/ folder not found: {DOCS_DIR}. Create it and add PDF files.")
        return

    for pdf in DOCS_DIR.glob("*.pdf"):
        print(f"[+] Loading {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} pages")
    if not docs:
        print("[!] No documents found to index.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print(f"🔹 Creating local embeddings (model: {EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("🔹 Building FAISS index (local)...")
    db = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    db.save_local(str(INDEX_DIR))
    print(f"✅ Local index built and saved to '{INDEX_DIR}/'")

def query_index(query: str):
    if not INDEX_DIR.exists():
        print(f"[ERROR] Index folder '{INDEX_DIR}' not found. Run: python basic_rag_retrieval.py index")
        return

    print("🔹 Loading FAISS index (local)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # Explicitly allow deserialization because we created the index locally.
    db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    results = retriever.get_relevant_documents(query)
    if not results:
        print("No relevant documents found.")
        return

    print("\n🔎 Top Results:\n")
    for i, r in enumerate(results, 1):
        src = r.metadata.get("source", "unknown")
        excerpt = r.page_content.strip().replace("\n", " ")[:800]
        print(f"{i}. Source: {src}\n{excerpt}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basic_rag_retrieval.py index|query \"your question\"")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "index":
        build_index()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Provide a query string.")
            sys.exit(1)
        query_text = " ".join(sys.argv[2:])
        query_index(query_text)
    else:
        print("Unknown command.")
