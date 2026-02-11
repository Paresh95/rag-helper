from typing import List
import asyncio
from src.utils.schemas import RAGResponse
from src.utils.schemas import RetrievedChunk
from src.online_components.retriever import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
)
from src.online_components.reranker import SentenceTransformerReranker
from src.online_components.generator import Generator, ChatMemory
from src.prompts.v1_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from qdrant_client import models as qm
from src.utils.vector_store import (
    client as qdrant_client,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


class VanillaRAG:
    """
    Implementation of simple RAG pipeline.
    """

    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        chunks = self.retriever.retrieve(query, self.k)
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks)
        return chunks

    async def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        return await self.generator.generate(query, chunks)

    async def run(self, retriever_query: str, generator_query: str) -> RAGResponse:
        nodes = self.retrieve(retriever_query)

        chunks: List[RetrievedChunk] = [
            RetrievedChunk(
                text=n.node.get_content(),
                score=float(n.score) if n.score is not None else 0.0,
                metadata=n.node.metadata or {},
            )
            for n in nodes
        ]

        answer = await self.generate_answer(generator_query, chunks)

        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            metadata={"num_contexts": len(chunks)},
        )


async def chat():
    filters = qm.Filter(
        must=[
            qm.FieldCondition(
                key="file_path",
                match=qm.MatchValue(value="data/processed/docling/2507.21110v1.json"),
            ),
        ]
    )

    dense_retriever = DenseRetriever(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        vector_name=DENSE_VECTOR_NAME,
        filters=filters,
    )

    sparse_retriever = SparseRetriever(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        vector_name=SPARSE_VECTOR_NAME,
        filters=filters,
    )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        method="weighted",
        weights=[0.5, 0.5],
        dense_k=10,
        sparse_k=10,
        hybrid_k=5,
    )

    reranker = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    )

    memory = ChatMemory(max_turns=5)

    generator = Generator(
        model_name="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        memory=memory,
    )

    rag = VanillaRAG(retriever, reranker, generator)

    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        response = await rag.run(query, query)

        print("Assistant:", response.answer)


if __name__ == "__main__":
    asyncio.run(chat())
    # query1 = "What does figure 2 show? It comes from: As shown in figure 2,
    # the semantic chunking algorithm works by first splitting... "
    # query2 = "What source is that from?"
