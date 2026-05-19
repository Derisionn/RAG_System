from typing import Any
from pydantic import BaseModel

class QueryResponse(BaseModel):
    question: str
    sql: str
    attempts: int
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int

class SQLOnlyResponse(BaseModel):
    question: str
    sql: str

class HealthResponse(BaseModel):
    status: str
    pinecone: str
    neo4j: str
    supabase: str
    gemini: str
