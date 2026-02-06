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


def upload_to_qdrant_fast(
    chunks: List[Chunk],
    vector_store: QdrantVectorStore,
    embedding_model: HuggingFaceEmbedding,
    batch_size: int = 256,
) -> bool:
    """
    Faster ingestion:
    - manual embedding
    - direct Qdrant upsert
    """

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]  # noqa: E203

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embedding_model.get_text_embedding_batch(texts)

        docs = [
            Document(text=t, metadata=m, embedding=e)
            for t, m, e in zip(texts, metadatas, embeddings)
        ]

        vector_store.add(docs)

    return True
