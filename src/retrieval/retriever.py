from utils.base import BaseRetriever
from utils.schemas import RetrievedChunk
from typing import List


# TODO: change name to HuggingFaceEmbeddingRetriever
class DenseRetriever(BaseRetriever):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.model.retrieve(query, k)


class SparseRetriever(BaseRetriever):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        return self.model.retrieve(query, k)


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
