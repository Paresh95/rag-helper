# vector_store.py
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
