from typing import Optional, Dict, Any
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models as qm

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

COLLECTION_NAME = "my_collection2"
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dense_dim = (
    dense_model.get_sentence_embedding_dimension()
)  # all-MiniLM-L6-v2 is 384 dims
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

client = QdrantClient(host="localhost", port=6333)
aclient = AsyncQdrantClient(host="localhost", port=6333)

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"


def ensure_collection(dense_dim: int):

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=dense_dim,
                distance=qm.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False)
            )
        },
    )


def _to_qdrant_filter(filters: Optional[Any]) -> Optional[qm.Filter]:
    """
    Accepts:
      - None
      - qm.Filter (already built)
      - dict in Qdrant filter shape (best-effort)
    """
    if filters is None:
        return None
    if isinstance(filters, qm.Filter):
        return filters
    if isinstance(filters, dict):

        def _cond_from_dict(d: Dict[str, Any]) -> qm.FieldCondition:
            key = d["key"]
            if "match" in d:
                m = d["match"]
                if "value" in m:
                    return qm.FieldCondition(
                        key=key, match=qm.MatchValue(value=m["value"])
                    )
                if "text" in m:
                    return qm.FieldCondition(
                        key=key, match=qm.MatchText(text=m["text"])
                    )
                if "any" in m:
                    return qm.FieldCondition(key=key, match=qm.MatchAny(any=m["any"]))
            if "range" in d:
                r = d["range"]
                return qm.FieldCondition(
                    key=key,
                    range=qm.Range(
                        gte=r.get("gte"),
                        gt=r.get("gt"),
                        lte=r.get("lte"),
                        lt=r.get("lt"),
                    ),
                )
            raise ValueError(f"Unsupported filter condition dict: {d}")

        must = [_cond_from_dict(c) for c in filters.get("must", [])]
        should = [_cond_from_dict(c) for c in filters.get("should", [])]
        must_not = [_cond_from_dict(c) for c in filters.get("must_not", [])]
        return qm.Filter(
            must=must or None, should=should or None, must_not=must_not or None
        )

    raise TypeError(
        "filters must be None, qdrant_client.models.Filter, or a dict in Qdrant filter shape"
    )
