from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(validation_alias="uuid", serialization_alias="uuid")
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def uuid(self) -> str:
        return self.chunk_id


class RAGResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
