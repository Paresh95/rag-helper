from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
