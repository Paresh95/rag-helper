from typing import List, Dict, Any
import uuid
import json

from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from src.utils.schemas import Chunk

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"

# choose a constant namespace UUID for your project (generate once and keep it)
NAMESPACE_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _stable_point_id(text: str, metadata: Dict[str, Any]) -> str:
    """
    Deterministic UUID derived from (text + metadata).
    Qdrant accepts UUID strings.
    """
    # Make metadata serialization stable (order + representation)
    meta_str = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    name = f"{text}\n{meta_str}"
    return str(uuid.uuid5(NAMESPACE_UUID, name))


def upload_to_qdrant(
    chunks: List[Chunk],
    client: QdrantClient,
    collection_name: str,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size: int = 256,
) -> bool:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]  # noqa: E203

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Dense embeddings (normalized for cosine)
        dense_vecs = dense_model.encode(
            texts,
            batch_size=min(64, len(texts)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Sparse embeddings (FastEmbed returns SparseEmbedding with indices/values)
        sparse_embs = list(sparse_model.embed(texts))

        points = []
        for t, m, dv, sv in zip(texts, metadatas, dense_vecs, sparse_embs):
            payload = dict(m)
            payload["text"] = t  # optional: keep original text in payload

            points.append(
                qm.PointStruct(
                    id=_stable_point_id(t, m),
                    vector={
                        DENSE_VECTOR_NAME: dv.tolist(),
                        SPARSE_VECTOR_NAME: qm.SparseVector(
                            indices=list(sv.indices),
                            values=list(sv.values),
                        ),
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    return True
