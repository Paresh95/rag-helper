import qdrant_client
from functools import lru_cache


@lru_cache(maxsize=1)
def get_qdrant_client(url: str = "http://localhost:6333") -> qdrant_client.QdrantClient:
    """
    Create a cached (singleton) Qdrant client.
    """

    return qdrant_client.QdrantClient(url=url)
