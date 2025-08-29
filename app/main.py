# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import shutil


from .models import IngestURL, AskRequest, AskResponse, Citation
from .ingest import ingest_pdfs, ingest_url
from .chains import build_qa_chain
from .models import IngestURL
from .ingest import ingest_pdfs, ingest_url

app = FastAPI(
    title="Smart Research Assistant API",
    description="An intelligent research assistant that ingests documents and provides semantic search capabilities.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# allow local Streamlit UI and other tools to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    """
    Welcome endpoint that provides basic information about the API.
    
    Returns:
        dict: Welcome message with API information
    """
    return {"message": "Welcome to the Smart Research Assistant API. Visit /docs for API documentation."}

@app.get("/health", tags=["General"])
def health():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: Status indicating the API health
    """
    return {"status": "ok"}

@app.post("/ingest", tags=["Document Ingestion"])
async def ingest(files: List[UploadFile] = File(default=[])):
    """
    Upload and ingest PDF documents into the vector database.
    
    This endpoint accepts multiple PDF files, saves them to the data directory,
    processes them into chunks, creates embeddings, and stores them in Chroma
    for semantic search capabilities.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        dict: List of successfully ingested file paths
        
    Example:
        Upload PDF files using form-data with field name "files"
    """
    saved_paths = []
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    for f in files:
        dest = data_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(str(dest))

    if saved_paths:
        ingest_pdfs(saved_paths)

    return {"ingested": saved_paths}

@app.post("/ingest_url", tags=["Document Ingestion"])
async def ingest_from_url(body: IngestURL):
    """
    Ingest content from a web URL into the vector database.
    
    This endpoint fetches content from the provided URL, processes it into chunks,
    creates embeddings, and stores them in Chroma for semantic search.
    
    Args:
        body: Request body containing the URL to ingest
        
    Returns:
        dict: Confirmation of the ingested URL
        
    Example:
        POST with JSON body: {"url": "https://example.com/article"}
    """
    ingest_url(body.url)
    return {"ingested_url": body.url}



@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    rag, retriever = build_qa_chain()
    answer = rag.invoke({"question": body.question})

    # collect top docs for structured citations
    docs = retriever.get_relevant_documents(body.question)
    citations: list[Citation] = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or d.metadata.get("url") or "unknown"
        page = d.metadata.get("page")
        snippet = d.page_content[:220]
        citations.append(Citation(source=str(src), page=page, snippet=snippet))

    return AskResponse(answer=answer, citations=citations, confidence=None)