from abc import ABC, abstractmethod
from typing import List
from utils.schemas import RetrievedChunk, Document


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[RetrievedChunk]:
        raise NotImplementedError("Must implement retrieve method")


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        raise NotImplementedError("Must implement generate method")


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, retrieved_chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        raise NotImplementedError("Must implement rerank method")


class BaseIngester(ABC):
    @abstractmethod
    def ingest(self, documents: List[Document]) -> None:
        raise NotImplementedError("Must implement ingest method")
