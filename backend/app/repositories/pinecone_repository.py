from pinecone import Pinecone
from ..config.config import PINECONE_API_KEY, PINECONE_INDEX

class PineconeRepository:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vector_index = self.pc.Index(PINECONE_INDEX)

    def query(self, vector: list[float], top_k: int = 20, include_metadata: bool = True, namespace: str = "", filter: dict = None):
        """Query Pinecone index for top_k similar vectors."""
        return self.vector_index.query(
            vector=vector, top_k=top_k, include_metadata=include_metadata, namespace=namespace, filter=filter
        )

    def upsert_history(self, vectors: list[dict], namespace: str):
        """Upsert embedded Q&A history vectors into a specific namespace."""
        self.vector_index.upsert(vectors=vectors, namespace=namespace)

    def describe_index_stats(self) -> dict:
        """Describe index stats for health checking."""
        return self.vector_index.describe_index_stats()
