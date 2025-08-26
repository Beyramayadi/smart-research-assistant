from pydantic import BaseModel, Field
from typing import List, Optional

class IngestURL(BaseModel):
    url: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class Citation(BaseModel):
    source: str
    page: int | None = None
    snippet: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float | None = None
