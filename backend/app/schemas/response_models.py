from typing import Any, Optional
from pydantic import BaseModel

class QueryResponse(BaseModel):
    session_id: str
    question: str
    sql: str
    attempts: int
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    answer: str | None = None
    chart_config: Optional[dict] = None

class SQLOnlyResponse(BaseModel):
    question: str
    sql: str

class HealthResponse(BaseModel):
    status: str
    pinecone: str
    neo4j: str
    supabase: str
    llm: str
    mongodb: str
