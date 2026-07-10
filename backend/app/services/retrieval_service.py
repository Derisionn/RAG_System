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
        from .cache_service import CacheService
        self.cache = CacheService()

    def retrieve_schema_elements(self, question: str) -> tuple[list[str], list[dict], list[list[str]], float, float]:
        """Perform full hybrid retrieval: vector search + plan filtering + Cypher join path extraction."""
        import time
        
        cache_key = self.cache.generate_key("vector_retrieval", question.strip().lower())
        cached_result = self.cache.get(cache_key)
        
        t0 = time.time()
        if cached_result:
            print("[RetrievalService] Vector Retrieval Cache HIT!")
            tables = cached_result["tables"]
            columns = cached_result["columns"]
            vector_ms = 0.0
        else:
            print(f"\n[RetrievalService] Embedding query: '{question}'")
            query_vec = self.embedding_srv.embed_text(question)
            results = self.pinecone_repo.query(query_vec, top_k=20)
            
            tables, columns = self.planner.plan_schema(results["matches"])
            
            self.cache.set(cache_key, {"tables": list(tables), "columns": columns}, ttl_seconds=3600 * 24)
            vector_ms = (time.time() - t0) * 1000

        # 4. Find join paths with GraphService
        t1 = time.time()
        paths = self.graph_srv.find_join_paths(tables)
        graph_ms = (time.time() - t1) * 1000

        return tables, columns, paths, vector_ms, graph_ms
