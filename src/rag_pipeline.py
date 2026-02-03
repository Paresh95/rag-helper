from typing import List

from src.utils.schemas import RAGResponse
from src.utils.schemas import RetrievedChunk


class VanillaRAG:
    """
    TODO: ADD DESCRIPTION
    """

    def __init__(self, retriever, reranker, generator, top_k):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        chunks = self.retriever.retrieve(query, self.top_k)
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks)
        return [
            RetrievedChunk(text=chunk.text, score=chunk.score, metadata=chunk.metadata)
            for chunk in chunks
        ]

    def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        return self.generator.generate(query, chunks)

    def run(self, query: str) -> RAGResponse:
        chunks = self.retrieve(query)
        answer = self.generate_answer(query, chunks)
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            metadata={
                "top_k": self.top_k,
                "num_contexts": len(chunks),
            },
        )

    # TODO: Implement batch processing
    # def run_batch(self, queries: List[str]) -> List[str]:
    #     return [self.run(query) for query in queries]

    # TODO: Implement async batch processing
    # def run_batch_async(self, queries: List[str]) -> List[str]:
    #     return asyncio.gather(*[self.run(query) for query in queries])
