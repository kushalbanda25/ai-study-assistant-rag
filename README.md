# 📘 AI Study Assistant (RAG + Local LLM)

## 🚀 Overview
An AI-powered study assistant that processes PDF documents and generates structured answers using Retrieval-Augmented Generation (RAG).

## ⚙️ Tech Stack
- Python
- Streamlit
- FAISS (Vector Search)
- Sentence Transformers
- Ollama (Phi Model - Local LLM)

## 🔥 Features
- Upload multiple PDFs
- Semantic search using FAISS
- GPT-like answers using local LLM
- Chat-based interface
- Persistent data storage

## 🧠 How it Works
1. Extract text from PDFs
2. Convert text into embeddings
3. Store embeddings in FAISS
4. Retrieve relevant chunks
5. Generate answers using LLM (Ollama)

## ▶️ Run Locally

```bash
pip install -r requirements.txt
ollama run phi
streamlit run app.py