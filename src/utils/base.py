from abc import ABC, abstractmethod
from typing import List
from src.utils.schemas import RetrievedChunk, Document


class BaseIngester(ABC):
    @abstractmethod
    def extract_data(self, raw_data: List[Document]) -> List[Document]:
        raise NotImplementedError("Must implement extract data method")

    @abstractmethod
    def process_data(self, extracted_data: List[Document]) -> List[Document]:
        raise NotImplementedError("Must implement process data method")

    @abstractmethod
    def chunk_data(self, processed_data: List[Document]) -> List[Document]:
        raise NotImplementedError("Must implement chunk data method")

    @abstractmethod
    def ingest_data(self, chunked_data: List[Document]) -> None:
        raise NotImplementedError("Must implement ingest data method")


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
