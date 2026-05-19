from pinecone import Pinecone
from ..config.config import PINECONE_API_KEY, PINECONE_INDEX

class PineconeRepository:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vector_index = self.pc.Index(PINECONE_INDEX)

    def query(self, vector: list[float], top_k: int = 20, include_metadata: bool = True):
        """Query Pinecone index for top_k similar vectors."""
        return self.vector_index.query(
            vector=vector, top_k=top_k, include_metadata=include_metadata
        )

    def describe_index_stats(self) -> dict:
        """Describe index stats for health checking."""
        return self.vector_index.describe_index_stats()
