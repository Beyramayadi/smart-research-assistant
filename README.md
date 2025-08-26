# Smart Research Assistant

A minimal **Research Assistant** powered by [LangChain](https://www.langchain.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/).  
The app can ingest PDFs or URLs, embed the text into a vector store, and answer questions with citations.

This project is built step by step as a learning journey into **RAG (Retrieval-Augmented Generation)** and **LLM application design**.

---

## Features (Week 1)

- Ingest PDFs and simple web pages.
- Store embeddings in a local [Chroma](https://www.trychroma.com/) vector database.
- Ask natural language questions against your ingested documents.
- Get grounded answers with source citations.
- Streamlit UI for a simple chat-like interface.

---

## Tech Stack

- **Backend**: FastAPI + LangChain
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: Chroma (local persistence)
- **Frontend**: Streamlit
- **Optional**: [Ollama](https://ollama.ai/) for local LLMs (e.g. Llama 3 Instruct)

---
