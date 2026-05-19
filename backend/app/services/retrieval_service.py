from ..repositories.pinecone_repository import PineconeRepository
from .embedding_service import EmbeddingService
from .graph_service import GraphService
from ..agents.planner_agent import PlannerAgent

class RetrievalService:
    def __init__(self, pinecone_repo: PineconeRepository, embedding_srv: EmbeddingService, graph_srv: GraphService, planner: PlannerAgent):
        self.pinecone_repo = pinecone_repo
        self.embedding_srv = embedding_srv
        self.graph_srv = graph_srv
        self.planner = planner

    def retrieve_schema_elements(self, question: str) -> tuple[list[str], list[dict], list[list[str]]]:
        """Perform full hybrid retrieval: vector search + plan filtering + Cypher join path extraction."""
        # 1. Embed question
        print(f"\n[RetrievalService] Embedding query: '{question}'")
        query_vec = self.embedding_srv.embed_text(question)

        # 2. Query Pinecone
        results = self.pinecone_repo.query(query_vec, top_k=20)

        # 3. Organize with Planner
        tables, columns = self.planner.plan_schema(results["matches"])

        # 4. Find join paths with GraphService
        paths = self.graph_srv.find_join_paths(tables)

        return tables, columns, paths
