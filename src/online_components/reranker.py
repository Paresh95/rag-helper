from typing import List
from sentence_transformers import CrossEncoder

from src.utils.base import BaseReranker
from src.utils.schemas import RetrievedChunk

from qdrant_client import models as qm

from src.utils.vector_store import (
    client as qdrant_client,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from src.online_components.retriever import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
)


class SentenceTransformerReranker(BaseReranker):
    def __init__(self, model_name: str, top_n: int):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(
        self, query: str, retrieved_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        if not retrieved_chunks:
            return []

        pairs = [(query, c.text) for c in retrieved_chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.top_n]

        reranked = []
        for chunk, score in ranked:
            score = float(score)
            chunk.score = score
            chunk.metadata["rerank_score"] = score
            reranked.append(chunk)

        return reranked


if __name__ == "__main__":
    filters = qm.Filter(
        must=[
            qm.FieldCondition(
                key="file_path",
                match=qm.MatchValue(value="data/processed/docling/2507.21110v1.json"),
            )
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
        rrf_k=60,
        hybrid_k=5,
    )

    query = """As shown in figure 2, the semantic chunking algorithm works by first splitting the
    input text into individual sentences, then encoding each sentence into a vector using a pre-trained
    language model. It calculates the cosine similarity between each sentence and the current chunk to
    determine semantic closeness. If the similarity is high, the sentence is grouped with the current chunk;
    otherwise, a new chunk is started. This results in contextually meaningful groups of sentences.
    These chunks can"""

    results = retriever.retrieve(query)
    reranker = SentenceTransformerReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
    )
    reranked = reranker.rerank(query, results)

    for i, chunk in enumerate(reranked, 1):
        print(f"\n#{i} score={chunk.score:.4f} uuid={chunk.uuid}")
        print(chunk.text[:500])
