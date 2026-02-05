from src.utils.schemas import Chunk
from typing import List
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def upload_to_qdrant(
    chunks: List[Chunk],
    vector_store: QdrantVectorStore,
    embedding_model: HuggingFaceEmbedding,
) -> bool:
    pipeline = IngestionPipeline(
        transformations=[
            embedding_model,
        ],
        vector_store=vector_store,
    )
    chunks_to_upload = []
    for chunk in chunks:
        chunks_to_upload.append(
            Document(
                text=chunk["text"],
                metadata=chunk["metadata"],
            )
        )
    pipeline.run(documents=chunks_to_upload)
    return True
