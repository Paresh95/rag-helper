from src.utils.base import BaseReranker
from src.utils.schemas import RetrievedChunk
from typing import List
from llama_index.core.postprocessor import SentenceTransformerRerank


class SentenceTransformerReranker(BaseReranker):
    def __init__(self, model_name: str, top_n: int):
        self.reranker = SentenceTransformerRerank(
            model=model_name,
            top_n=top_n,
        )

    def rerank(
        self, query: str, retrieved_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        reranked_chunks = self.reranker.postprocess_nodes(
            retrieved_chunks, query_str=query
        )
        for chunk in reranked_chunks:
            chunk.metadata["rerank_score"] = chunk.score  # Preserve rerank score
        return reranked_chunks
