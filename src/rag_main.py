from typing import List

from src.utils.schemas import RAGResponse
from src.utils.schemas import RetrievedChunk
from src.online_components.retriever import SimpleHybridRetriever
from src.online_components.reranker import SentenceTransformerReranker
from src.online_components.generator import Generator
from src.prompts.v1_prompt import USER_PROMPT_TEMPLATE


class VanillaRAG:
    """
    Implementation of simple RAG pipeline.
    """

    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        chunks = self.retriever.retrieve(query)
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks)
        return chunks

    def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        return self.generator.generate(query, chunks)

    def run(self, retriever_query: str, generator_query: str) -> RAGResponse:
        nodes = self.retrieve(retriever_query)
        chunks: List[RetrievedChunk] = [
            RetrievedChunk(
                text=n.node.get_content(),
                score=float(n.score) if n.score is not None else 0.0,
                metadata=n.node.metadata or {},
            )
            for n in nodes
        ]
        answer = self.generate_answer(generator_query, chunks)
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            metadata={
                "num_contexts": len(chunks),
            },
        )


if __name__ == "__main__":
    retriever = SimpleHybridRetriever(
        similarity_top_k=10, sparse_top_k=10, hybrid_top_k=7
    )
    reranker = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3
    )
    generator = Generator(
        model_name="gpt-4o-mini", prompt_template=USER_PROMPT_TEMPLATE
    )

    retriever_query = (
        "As shown in figure 2, the semantic chunking algorithm works by first splitting"
    )
    generator_query = "What does figure 2 show?"

    rag_pipeline = VanillaRAG(retriever, reranker, generator)
    response = rag_pipeline.run(retriever_query, generator_query)
    print(response)
