from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository

class HealthController:
    def __init__(self, rag_service: RAGService, mongo_repo: MongoRepository):
        self.rag_service = rag_service
        self.mongo_repo = mongo_repo

    def get_health_status(self) -> dict:
        """Run health check validations against all backend components."""
        results = {
            "pinecone": "unknown",
            "neo4j": "unknown",
            "supabase": "unknown",
            "llm": "unknown",
            "mongodb": "unknown",
        }

        # Pinecone
        try:
            stats = self.rag_service.pinecone_repo.describe_index_stats()
            results["pinecone"] = f"ok - {stats.get('total_vector_count', '?')} vectors"
        except Exception as e:
            results["pinecone"] = f"error: {str(e)[:120]}"

        # Neo4j
        try:
            count = self.rag_service.graph_srv.count_nodes()
            results["neo4j"] = f"ok - {count} table nodes"
        except Exception as e:
            results["neo4j"] = f"error: {str(e)[:120]}"

        # Supabase (PostgreSQL)
        try:
            self.rag_service.postgres_repo.check_connection()
            results["supabase"] = "ok"
        except Exception as e:
            results["supabase"] = f"error: {str(e)[:120]}"

        # LLM (Hugging Face)
        try:
            resp_text = self.rag_service.reasoner.generate_sql("Reply with: ok")
            results["llm"] = "ok" if resp_text else "no response"
        except Exception as e:
            results["llm"] = f"error: {str(e)[:120]}"

        # MongoDB
        try:
            self.mongo_repo.check_connection()
            results["mongodb"] = "ok"
        except Exception as e:
            results["mongodb"] = f"error: {str(e)[:120]}"

        overall = (
            "healthy"
            if all(v.startswith("ok") for v in results.values())
            else "degraded"
        )

        return {"status": overall, **results}
