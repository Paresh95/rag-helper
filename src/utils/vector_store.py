from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore


dense_embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    normalize=True,  # cosine similarity friendly
    embed_batch_size=16,  # tune for your hardware
)

Settings.embed_model = dense_embedding_model

client = QdrantClient(host="localhost", port=6333)
aclient = AsyncQdrantClient(host="localhost", port=6333)

qdrant_vector_store = QdrantVectorStore(
    collection_name="my_collection",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    batch_size=20,  # controls sparse batch processing
    fastembed_sparse_model="prithivida/Splade_PP_en_v1",
)
