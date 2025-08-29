# app/ingest.py
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from .config import settings, VECTOR_DIR

# How to split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=160
)

def build_embeddings():
    """Return the embedding model."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def load_and_split_pdf(path: str):
    """Load a PDF and split it into chunks."""
    docs = PyPDFLoader(path).load()
    return splitter.split_documents(docs)

def load_and_split_url(url: str):
    """Load a web page and split into chunks."""
    docs = WebBaseLoader(url).load()
    return splitter.split_documents(docs)

def get_vectorstore():
    """Return (or create) the persistent Chroma vector store."""
    embeddings = build_embeddings()
    vs = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR)
    )
    return vs

def ingest_pdfs(paths: list[str]):
    """Ingest multiple PDF files into Chroma."""
    vs = get_vectorstore()
    for p in paths:
        chunks = load_and_split_pdf(p)
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vs.add_texts(texts=texts, metadatas=metadatas)
    vs.persist()

def ingest_url(url: str):
    """Ingest a single URL into Chroma."""
    vs = get_vectorstore()
    chunks = load_and_split_url(url)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vs.add_texts(texts=texts, metadatas=metadatas)
    vs.persist()
