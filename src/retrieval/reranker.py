from utils.base import BaseReranker
from utils.schemas import RetrievedChunk
from typing import List


class SentenceTransformerReranker(BaseReranker):  # TODO: Change name
    def __init__(self, model_name: str, top_n: int):
        self.model_name = model_name
        self.top_n = top_n

    def rerank(
        self, query: str, retrieved_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        return self.model.rerank(query, retrieved_chunks)
