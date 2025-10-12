# Task-03: Local RAG Retrieval System

This project implements a **local Retrieval-Augmented Generation (RAG)** system using **FAISS** and **HuggingFace embeddings**. It allows you to query PDF documents locally without relying on external APIs.

## Features
- Load and index local PDF documents.
- Split documents into chunks for efficient retrieval.
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Build and query a FAISS vector store locally.
- Retrieve top relevant document excerpts for a given query.

