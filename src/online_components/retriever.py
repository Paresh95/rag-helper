from src.utils.base import BaseRetriever
from src.utils.schemas import RetrievedChunk
from typing import List
from src.utils.vector_store import dense_embedding_model, qdrant_vector_store
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex


def simple_hybrid_retriever(
    query: str, similarity_top_k: int, sparse_top_k: int, hybrid_top_k: int
) -> List[RetrievedChunk]:
    index = VectorStoreIndex.from_vector_store(
        vector_store=qdrant_vector_store,
        embed_model=dense_embedding_model,
    )
    base_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        sparse_top_k=sparse_top_k,
        vector_store_query_mode="hybrid",
        hybrid_top_k=hybrid_top_k,
    )
    initial_nodes = base_retriever.retrieve(query)

    for n in initial_nodes:  # Preserve similarity scores
        n.node.metadata["similarity_score"] = n.score

    return initial_nodes


# TODO: Add base model for hugging face

# TODO: add custom model for dense retrieval
class DenseRetriever(BaseRetriever):
    def __init__(self, embedding_model: HuggingFaceEmbedding):
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.model.retrieve(query, k)


# TODO: Add custom model for sparse retrieval
class SparseRetriever(BaseRetriever):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.model.retrieve(query, k)


# TODO: Add custom model for hybrid retrieval
class HybridRetriever(BaseRetriever):

    """
    TODO: Implement
    Methods: [weighted, reciprocal_rank]
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        method: str,
        weights: List[float],
        dense_k: int,
        sparse_k: int,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.method = method
        self.weights = weights
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        dense_chunks = self.dense_retriever.retrieve(query, self.dense_k)
        sparse_chunks = self.sparse_retriever.retrieve(query, self.sparse_k)
        return [*dense_chunks, *sparse_chunks]
