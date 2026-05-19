from ..services.rag_service import RAGService

class HealthController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def get_health_status(self) -> dict:
        """Run health check validations against all backend components."""
        results = {
            "pinecone": "unknown",
            "neo4j": "unknown",
            "supabase": "unknown",
            "gemini": "unknown",
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

        # Gemini
        try:
            resp_text = self.rag_service.reasoner.generate_sql("Reply with: ok")
            results["gemini"] = "ok" if resp_text else "no response"
        except Exception as e:
            results["gemini"] = f"error: {str(e)[:120]}"

        overall = (
            "healthy"
            if all(v.startswith("ok") for v in results.values())
            else "degraded"
        )

        return {"status": overall, **results}
