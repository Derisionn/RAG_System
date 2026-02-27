"""
api.py
──────
FastAPI layer for the Agentic Hybrid SQL RAG System.
Exposes the RAG pipeline as a REST API.

Endpoints:
  POST /query          — Natural language → SQL → Execute → Return results
  POST /query/sql-only — Natural language → SQL only (no execution)
  GET  /health         — Health check for all connected services

Run:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import SQLRAGPipeline
from config import MAX_RETRIES


# ── Lifespan: initialize pipeline once at startup ────────────────────────────
pipeline: SQLRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: create shared pipeline. Shut-down: close connections."""
    global pipeline
    print("[startup] Initializing SQLRAGPipeline...")
    pipeline = SQLRAGPipeline()
    print("[startup] Pipeline ready ✅")
    yield
    print("[shutdown] Closing pipeline connections...")
    if pipeline:
        pipeline.close()
    print("[shutdown] Done.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SQL RAG System API",
    description=(
        "Agentic Hybrid RAG system that converts natural language questions "
        "into T-SQL queries for the AdventureWorks2019 database, executes them, "
        "and returns the results — with automatic self-correction on failure."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="Who are the top 5 customers by total sales amount?",
        description="Natural language question about the AdventureWorks database.",
    )


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
    mssql: str
    gemini: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Check connectivity to all backend services:
    Pinecone, Neo4j, MSSQL, and Gemini.
    """
    results: dict[str, str] = {
        "pinecone": "unknown",
        "neo4j": "unknown",
        "mssql": "unknown",
        "gemini": "unknown",
    }

    # Pinecone
    try:
        stats = pipeline.vector_index.describe_index_stats()
        results["pinecone"] = f"ok — {stats.get('total_vector_count', '?')} vectors"
    except Exception as e:
        results["pinecone"] = f"error: {str(e)[:120]}"

    # Neo4j
    try:
        with pipeline.graph_driver.session() as session:
            count = session.run("MATCH (t:Table) RETURN count(t) AS n").single()["n"]
        results["neo4j"] = f"ok — {count} table nodes"
    except Exception as e:
        results["neo4j"] = f"error: {str(e)[:120]}"

    # MSSQL
    try:
        with pipeline.sql_engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        results["mssql"] = "ok"
    except Exception as e:
        results["mssql"] = f"error: {str(e)[:120]}"

    # Gemini
    try:
        resp = pipeline.llm.generate_content("Reply with: ok")
        results["gemini"] = "ok" if resp.text else "no response"
    except Exception as e:
        results["gemini"] = f"error: {str(e)[:120]}"

    overall = (
        "healthy"
        if all(v.startswith("ok") for v in results.values())
        else "degraded"
    )

    return HealthResponse(status=overall, **results)


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Full RAG pipeline:
    1. Semantic retrieval from Pinecone
    2. Join-path retrieval from Neo4j
    3. Prompt assembly
    4. SQL generation via Gemini (with self-correction, up to MAX_RETRIES)
    5. SQL execution on MSSQL

    Returns the generated SQL, result rows, and number of attempts used.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        sql, df, error = pipeline.generate_sql(request.question)
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

    # Count attempts (simple heuristic from the pipeline — TODO: surface from pipeline)
    columns = list(df.columns)
    rows = df.head(100).to_dict(orient="records")  # cap rows at 100

    return QueryResponse(
        question=request.question,
        sql=sql,
        attempts=1,          # pipeline will surface this later
        columns=columns,
        rows=rows,
        row_count=len(df),
    )


@app.post("/query/sql-only", response_model=SQLOnlyResponse, tags=["RAG"])
def query_sql_only(request: QueryRequest):
    """
    Generate SQL without executing it.
    Useful for preview / debugging.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        tables, columns = pipeline.retrieve_schema_elements(request.question)
        paths = pipeline.get_join_paths(tables)
        prompt = pipeline.build_prompt(request.question, tables, columns, paths)

        import google.generativeai as genai  # already configured
        response = pipeline.llm.generate_content(prompt)
        sql = response.text.strip().replace("```sql", "").replace("```", "").strip()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Generation error:\n{traceback.format_exc()}",
        )

    return SQLOnlyResponse(question=request.question, sql=sql)
