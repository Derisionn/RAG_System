import traceback
from fastapi import HTTPException
from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository
from ..config.config import MAX_RETRIES

# Google API exception — imported defensively so we can catch quota errors
try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
except ImportError:
    _ResourceExhausted = None


def _is_quota_error(exc: Exception) -> bool:
    """Return True if exc is a Gemini rate-limit / quota-exhausted error."""
    if _ResourceExhausted and isinstance(exc, _ResourceExhausted):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("resourceexhausted", "quota exceeded", "429", "rate limit", "resource_exhausted"))


_QUOTA_DETAIL = {
    "message": "Gemini API free-tier daily quota exceeded (limit: 20 requests/day).",
    "action": (
        "Wait until your quota resets (usually midnight Pacific Time) "
        "or upgrade to a pay-as-you-go Gemini API plan at "
        "https://ai.google.dev/gemini-api/docs/rate-limits"
    ),
}


class ChatController:
    def __init__(self, rag_service: RAGService, mongo_repo: MongoRepository):
        self.rag_service = rag_service
        self.mongo_repo = mongo_repo

    def execute_query(self, question: str, session_id: str) -> dict:
        """Coordinate with RAG service to run query and return formatted results."""
        # 1. Load conversation history from MongoDB
        history = []
        try:
            history = self.mongo_repo.get_history(session_id)
        except Exception as e:
            print(f"[ChatController] Warning: could not load history: {e}")

        try:
            sql, df, error = self.rag_service.execute_rag(question, history=history)
        except Exception as exc:
            if _is_quota_error(exc):
                raise HTTPException(status_code=429, detail=_QUOTA_DETAIL)
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error:\n{traceback.format_exc()}",
            )

        if error:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"SQL generation failed after {MAX_RETRIES} attempts.",
                    "last_sql": sql,
                    "error": error[:500],
                },
            )

        columns = list(df.columns)
        rows = df.head(100).to_dict(orient="records")

        # 2. Save this Q&A turn to MongoDB
        try:
            self.mongo_repo.save_message(session_id, question, sql, rows)
        except Exception as e:
            print(f"[ChatController] Warning: could not save to MongoDB: {e}")

        return {
            "session_id": session_id,
            "question": question,
            "sql": sql,
            "attempts": self.rag_service.last_attempts,
            "columns": columns,
            "rows": rows,
            "row_count": len(df),
        }

    def generate_sql_only(self, question: str) -> dict:
        """Generate SQL query for review without executing it."""
        try:
            tables, columns = self.rag_service.retrieve_schema_elements_only(question)
            paths = self.rag_service.graph_srv.find_join_paths(tables)
            prompt = self.rag_service.reasoner.build_prompt(question, tables, columns, paths)
            sql = self.rag_service.reasoner.generate_sql(prompt)
            return {"question": question, "sql": sql}
        except Exception as exc:
            if _is_quota_error(exc):
                raise HTTPException(status_code=429, detail=_QUOTA_DETAIL)
            raise HTTPException(
                status_code=500,
                detail=f"Generation error:\n{traceback.format_exc()}",
            )
