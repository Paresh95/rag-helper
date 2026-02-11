# retriever.py
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.utils.base import BaseRetriever
from src.utils.schemas import RetrievedChunk

from src.utils.vector_store import (
    client as qdrant_client,
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    dense_model,
    sparse_model,
    _to_qdrant_filter,
)


def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _make_chunk(point: qm.ScoredPoint) -> RetrievedChunk:
    """
    Convert a Qdrant ScoredPoint into RetrievedChunk schema format.
    """

    payload = point.payload or {}
    chunk_id = str(payload.get("uuid") or point.id)

    # Extract text; everything else becomes metadata
    text = payload.get("text", "")
    metadata = dict(payload)
    metadata.pop("text", None)  # keep text only in .text

    score = float(point.score)
    metadata["similarity_score"] = score

    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=metadata,
    )


class DenseRetriever(BaseRetriever):
    """
    Dense vector retriever using SentenceTransformers over Qdrant using the named dense vector.
    """

    def __init__(
        self,
        client: QdrantClient = qdrant_client,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = DENSE_VECTOR_NAME,
        filters: Optional[Any] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.filters = filters

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        q_filter = _to_qdrant_filter(self.filters)

        query_vec = dense_model.encode(query, normalize_embeddings=True).tolist()

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=qm.NearestQuery(nearest=query_vec),
            using=self.vector_name,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return [_make_chunk(p) for p in resp.points]


class SparseRetriever(BaseRetriever):
    """
    Sparse retriever using fastembed SparseTextEmbedding to generate SPLADE sparse vectors.
    """

    def __init__(
        self,
        client: QdrantClient = qdrant_client,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = SPARSE_VECTOR_NAME,
        filters: Optional[Any] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.filters = filters

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        q_filter = _to_qdrant_filter(self.filters)

        emb = next(iter(sparse_model.embed([query])))
        indices = list(getattr(emb, "indices"))
        values = list(getattr(emb, "values"))

        sparse_vec = qm.SparseVector(indices=indices, values=values)

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=qm.NearestQuery(nearest=sparse_vec),
            using=self.vector_name,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return [_make_chunk(p) for p in resp.points]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that fuses DenseRetriever + SparseRetriever.

    Parameters:
        -  method:
            - "weighted": weighted sum of min-max normalized scores
            - "reciprocal_rank": Reciprocal Rank Fusion (RRF)
        - weights: [dense_weight, sparse_weight] -> can be used for weighted or RRF methods
        - dense_k/sparse_k/hybrid_k are candidate pool sizes (like similarity_top_k and sparse_top_k)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        method: str = "reciprocal_rank",
        weights: Optional[List[float]] = None,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 60,  # standard smoothing parameter
        hybrid_k: int = 10,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.method = method
        self.weights = weights or [0.5, 0.5]
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.hybrid_k = hybrid_k

        self.rrf_k = rrf_k

        if self.method not in {"weighted", "reciprocal_rank"}:
            raise ValueError("method must be one of {'weighted', 'reciprocal_rank'}")
        if len(self.weights) != 2:
            raise ValueError(
                "weights must have length 2: [dense_weight, sparse_weight]"
            )

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        - retrieve(query, k) returns the final top-k (like hybrid_top_k)
        """
        dense_chunks = self.dense_retriever.retrieve(query, self.dense_k)
        sparse_chunks = self.sparse_retriever.retrieve(query, self.sparse_k)

        if self.method == "weighted":
            return self._fuse_weighted(dense_chunks, sparse_chunks, self.hybrid_k)

        return self._fuse_weighted_rrf(dense_chunks, sparse_chunks, self.hybrid_k)

    def _fuse_weighted(
        self,
        dense_chunks: List[RetrievedChunk],
        sparse_chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        dw, sw = self.weights
        fused: Dict[
            str, Tuple[RetrievedChunk, float]
        ] = {}  # chunk_id -> (best_chunk_obj, fused_score)

        def accumulate_weighted_scores(chunks: List[RetrievedChunk], weight: float):
            scores = [c.score for c in chunks]
            norm_scores = _minmax_normalize(scores)
            for c, ns in zip(chunks, norm_scores):
                fused_score = weight * ns
                chunk, current_score = fused.get(c.chunk_id, (c, 0.0))
                fused[c.chunk_id] = (chunk, current_score + fused_score)

        accumulate_weighted_scores(dense_chunks, dw)
        accumulate_weighted_scores(sparse_chunks, sw)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[:k]

        out: List[RetrievedChunk] = []
        for chunk, score in ranked:
            chunk.score = float(score)
            chunk.metadata["fused_similarity_score"] = float(score)
            out.append(chunk)
        return out

    def _fuse_weighted_rrf(
        self,
        dense_chunks: List[RetrievedChunk],
        sparse_chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        dw, sw = self.weights
        fused: Dict[str, Tuple[RetrievedChunk, float]] = {}

        def accumulate_weighted_rrf_scores(
            chunks: List[RetrievedChunk], weight: float
        ) -> None:
            for rank, c in enumerate(chunks, start=1):
                score = weight * (1.0 / (self.rrf_k + rank))
                chunk, current_score = fused.get(c.chunk_id, (c, 0.0))
                fused[c.chunk_id] = (chunk, current_score + score)

        accumulate_weighted_rrf_scores(dense_chunks, dw)
        accumulate_weighted_rrf_scores(sparse_chunks, sw)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[:k]

        out: List[RetrievedChunk] = []
        for chunk, score in ranked:
            chunk.score = float(score)
            chunk.metadata["fused_similarity_score"] = float(score)
            out.append(chunk)
        return out


if __name__ == "__main__":
    filters = qm.Filter(
        must=[
            qm.FieldCondition(
                key="file_path",
                match=qm.MatchValue(value="data/processed/docling/2507.21110v1.json"),
            ),
            # qm.FieldCondition(
            #     key="page_no",
            #     range=qm.Range(gte=5, lte=10),
            # ),
        ]
    )

    # Create dense + sparse retrievers with the same filter
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

    results = retriever.retrieve(
        """As shown in figure 2, the semantic chunking algorithm works by first splitting the
        input text into individual sentences, then encoding each sentence into a vector using a pre-trained
        language model. It calculates the cosine similarity between each sentence and the current chunk to
        determine semantic closeness. If the similarity is high, the sentence is grouped with the current chunk;
        otherwise, a new chunk is started. This results in contextually meaningful groups of sentences.
        These chunks can"""
    )

    # Optional: print results
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.6f} chunk_id={r.chunk_id}")
        print(f"metadata={r.metadata}")
        print(r.text[:500], "..." if len(r.text) > 500 else "")
