import traceback
from fastapi import HTTPException
from ..services.rag_service import RAGService
from ..config.config import MAX_RETRIES

class ChatController:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def execute_query(self, question: str) -> dict:
        """Coordinate with RAG service to run query and return formatted results."""
        try:
            sql, df, error = self.rag_service.execute_rag(question)
        except Exception:
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

        return {
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
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Generation error:\n{traceback.format_exc()}",
            )
