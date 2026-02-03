from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any]
