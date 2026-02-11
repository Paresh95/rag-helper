import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient, models as qm

from src.utils.base import BaseRetriever
from src.utils.schemas import RetrievedChunk

from src.utils.vector_store import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    dense_model,
    sparse_model,
    _to_qdrant_filter,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _make_chunk(point: qm.ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    chunk_id = str(payload.get("uuid") or point.id)

    text = payload.get("text", "")
    metadata = dict(payload)
    metadata.pop("text", None)

    score = float(point.score)
    metadata["similarity_score"] = score

    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Async Dense Retriever
# -----------------------------------------------------------------------------


class AsyncDenseRetriever(BaseRetriever):
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = DENSE_VECTOR_NAME,
        filters: Optional[Any] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.filters = filters

    async def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        q_filter = _to_qdrant_filter(self.filters)

        # Embedding is CPU-bound → run in thread so event loop is not blocked
        query_vec = await asyncio.to_thread(
            dense_model.encode, query, normalize_embeddings=True
        )

        resp = await self.client.query_points(
            collection_name=self.collection_name,
            query=qm.NearestQuery(nearest=query_vec.tolist()),
            using=self.vector_name,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        return [_make_chunk(p) for p in resp.points]


# -----------------------------------------------------------------------------
# Async Sparse Retriever
# -----------------------------------------------------------------------------


class AsyncSparseRetriever(BaseRetriever):
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = COLLECTION_NAME,
        vector_name: str = SPARSE_VECTOR_NAME,
        filters: Optional[Any] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.filters = filters

    async def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        q_filter = _to_qdrant_filter(self.filters)

        emb = await asyncio.to_thread(lambda: next(iter(sparse_model.embed([query]))))

        sparse_vec = qm.SparseVector(
            indices=list(getattr(emb, "indices")),
            values=list(getattr(emb, "values")),
        )

        resp = await self.client.query_points(
            collection_name=self.collection_name,
            query=qm.NearestQuery(nearest=sparse_vec),
            using=self.vector_name,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        return [_make_chunk(p) for p in resp.points]


# -----------------------------------------------------------------------------
# Async Hybrid Retriever (parallel dense + sparse)
# -----------------------------------------------------------------------------


class AsyncHybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: AsyncDenseRetriever,
        sparse_retriever: AsyncSparseRetriever,
        method: str = "reciprocal_rank",
        weights: Optional[List[float]] = None,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 60,
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

    async def retrieve(self, query: str) -> List[RetrievedChunk]:
        # 🔑 Run dense + sparse concurrently
        dense_chunks, sparse_chunks = await asyncio.gather(
            self.dense_retriever.retrieve(query, self.dense_k),
            self.sparse_retriever.retrieve(query, self.sparse_k),
        )

        if self.method == "weighted":
            return self._fuse_weighted(dense_chunks, sparse_chunks, self.hybrid_k)

        return self._fuse_weighted_rrf(dense_chunks, sparse_chunks, self.hybrid_k)

    # ------------------------------------------------------------------
    # Fusion methods (same as your sync logic)
    # ------------------------------------------------------------------

    def _fuse_weighted(
        self,
        dense_chunks: List[RetrievedChunk],
        sparse_chunks: List[RetrievedChunk],
        k: int,
    ) -> List[RetrievedChunk]:
        dw, sw = self.weights
        fused: Dict[str, Tuple[RetrievedChunk, float]] = {}

        def accumulate(chunks: List[RetrievedChunk], weight: float):
            scores = [c.score for c in chunks]
            norm_scores = _minmax_normalize(scores)
            for c, ns in zip(chunks, norm_scores):
                fused_score = weight * ns
                chunk, current = fused.get(c.chunk_id, (c, 0.0))
                fused[c.chunk_id] = (chunk, current + fused_score)

        accumulate(dense_chunks, dw)
        accumulate(sparse_chunks, sw)

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

        def accumulate(chunks: List[RetrievedChunk], weight: float):
            for rank, c in enumerate(chunks, start=1):
                score = weight * (1.0 / (self.rrf_k + rank))
                chunk, current = fused.get(c.chunk_id, (c, 0.0))
                fused[c.chunk_id] = (chunk, current + score)

        accumulate(dense_chunks, dw)
        accumulate(sparse_chunks, sw)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[:k]

        out: List[RetrievedChunk] = []
        for chunk, score in ranked:
            chunk.score = float(score)
            chunk.metadata["fused_similarity_score"] = float(score)
            out.append(chunk)
        return out


# -----------------------------------------------------------------------------
# FastAPI integration (production usage)
# -----------------------------------------------------------------------------


def create_app(qdrant_url: str = "http://localhost:6333") -> FastAPI:
    app = FastAPI()

    client = AsyncQdrantClient(url=qdrant_url)

    dense = AsyncDenseRetriever(client=client)
    sparse = AsyncSparseRetriever(client=client)

    hybrid = AsyncHybridRetriever(dense, sparse)

    @app.get("/search")
    async def search(query: str, k: int = 5):
        results = await hybrid.retrieve(query)
        return [r.model_dump() for r in results[:k]]

    return app


app = create_app()
