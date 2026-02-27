"""
rag_pipeline.py
───────────────
Agentic Hybrid SQL RAG System powered by LangGraph.

Graph nodes:
  retrieve      → Semantic (Pinecone) + Relational (Neo4j) retrieval
  generate_sql  → Gemini LLM generates or corrects SQL
  validate_sql  → Safety + syntax pre-check (SELECT-only, sqlparse)
  execute_sql   → Run SQL on MSSQL via SQLAlchemy

Flow:
  START → retrieve → generate_sql → validate_sql ──→ execute_sql → END
                          ↑           (invalid)            |
                          └────────── correction ──────────┘ (on error)
"""

from __future__ import annotations

import os
import pandas as pd
import sqlparse
from typing import TypedDict, Optional

from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import google.generativeai as genai

from langgraph.graph import StateGraph, END

from .config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PWD,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    CONNECTION_STRING,
    TOP_K_TABLES,
    TOP_K_COLUMNS,
    MAX_RETRIES,
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between all graph nodes."""
    question: str
    tables: list[str]
    columns: list[dict]
    paths: list[list[str]]
    prompt: str
    sql: str
    error: Optional[str]
    result: Optional[object]   # pandas DataFrame or None
    attempts: int
    validation_error: Optional[str]


# ── SQLRAGPipeline ────────────────────────────────────────────────────────────

class SQLRAGPipeline:
    """
    Wraps the LangGraph agent as a clean Python object.
    External interface (api.py) is unchanged:
        sql, df, error = pipeline.generate_sql(question)
    """

    def __init__(self):
        print("Initializing Retrieval Engine (Pinecone + Neo4j + LangGraph)...")

        # 1. Embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # 2. Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vector_index = self.pc.Index(PINECONE_INDEX)

        # 3. Neo4j
        self.graph_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD)
        )

        # 4. Gemini
        self.llm = genai.GenerativeModel(GEMINI_MODEL)

        # 5. SQL engine
        self.sql_engine = create_engine(CONNECTION_STRING)

        # 6. Build and compile the LangGraph agent
        self.agent = self._build_graph()
        print("LangGraph agent compiled ✅")

    def close(self):
        self.graph_driver.close()

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """
        Node 1 — Retrieve
        Semantic search via Pinecone + join-path search via Neo4j.
        """
        question = state["question"]
        print(f"\n[Node: retrieve] Embedding query: '{question}'")

        # Semantic retrieval
        query_vec = self.model.encode([question]).tolist()[0]
        results = self.vector_index.query(
            vector=query_vec, top_k=20, include_metadata=True
        )

        tables: set[str] = set()
        columns: list[dict] = []
        for res in results["matches"]:
            meta = res["metadata"]
            score = res["score"]
            if meta["type"] == "table":
                tables.add(meta["table"])
                print(f"  ▸ Table: {meta['table']} (score={score:.4f})")
            else:
                columns.append(meta)
                tables.add(meta["table"])
                print(f"  ▸ Column: {meta['table']}.{meta['column']} (score={score:.4f})")
            if len(tables) >= TOP_K_TABLES and len(columns) >= TOP_K_COLUMNS:
                break

        tables_list = list(tables)

        # Relational retrieval
        print(f"\n[Node: retrieve] Finding join paths between {len(tables_list)} tables...")
        paths: list[list[str]] = []
        cypher = """
        MATCH (a:Table {name: $start_node}), (b:Table {name: $end_node})
        MATCH p = shortestPath((a)-[:REFERENCES*..3]-(b))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """
        if len(tables_list) >= 2:
            with self.graph_driver.session() as session:
                for i in range(len(tables_list)):
                    for j in range(i + 1, len(tables_list)):
                        result = session.run(
                            cypher,
                            start_node=tables_list[i],
                            end_node=tables_list[j],
                        )
                        for record in result:
                            path = record["path_nodes"]
                            if path and len(path) > 1:
                                paths.append(path)
                                print(f"  ▸ Path: {' -> '.join(path)}")

        return {**state, "tables": tables_list, "columns": columns, "paths": paths}

    def _node_generate_sql(self, state: AgentState) -> AgentState:
        """
        Node 2 — Generate / Correct SQL
        Builds the prompt and calls Gemini. On first attempt uses the base
        prompt; on subsequent attempts includes the error for self-correction.
        """
        attempt = state["attempts"] + 1
        print(f"\n[Node: generate_sql] Calling Gemini (attempt {attempt})...")

        if state["attempts"] == 0:
            # First attempt — build base prompt
            prompt = self._build_prompt(
                state["question"], state["tables"], state["columns"], state["paths"]
            )
        else:
            # Correction attempt — include previous SQL + error
            prompt = self._build_correction_prompt(
                state["prompt"], state["sql"], state["error"] or state["validation_error"] or ""
            )

        response = self.llm.generate_content(prompt)
        sql = response.text.strip().replace("```sql", "").replace("```", "").strip()
        print(f"  ▸ SQL generated:\n{sql[:300]}")

        return {
            **state,
            "prompt": prompt,
            "sql": sql,
            "attempts": attempt,
            "error": None,
            "validation_error": None,
        }

    def _node_validate_sql(self, state: AgentState) -> AgentState:
        """
        Node 3 — Validate SQL
        1. Safety guard — only SELECT statements are allowed.
        2. Syntax check — sqlparse ensures the query is parseable.
        """
        sql = state["sql"]
        print(f"\n[Node: validate_sql] Checking SQL safety and syntax...")

        # 1. Safety guard — strip comments and whitespace then check keyword
        parsed = sqlparse.parse(sql)
        if not parsed:
            return {**state, "validation_error": "Empty SQL returned by LLM."}

        first_token = parsed[0].get_type()
        if first_token is None or first_token.upper() != "SELECT":
            msg = (
                f"Safety violation: only SELECT queries are allowed. "
                f"Got statement type: '{first_token}'."
            )
            print(f"  ✗ {msg}")
            return {**state, "validation_error": msg}

        # 2. Basic syntax check — ensure sqlparse can parse at least one token
        stmt = parsed[0]
        tokens = [t for t in stmt.tokens if not t.is_whitespace]
        if len(tokens) < 2:
            msg = "SQL appears malformed — too few tokens."
            print(f"  ✗ {msg}")
            return {**state, "validation_error": msg}

        print("  ✓ SQL passed safety and syntax checks.")
        return {**state, "validation_error": None}

    def _node_execute_sql(self, state: AgentState) -> AgentState:
        """
        Node 4 — Execute SQL
        Runs the validated SQL on MSSQL and returns a DataFrame.
        """
        sql = state["sql"]
        print(f"\n[Node: execute_sql] Running SQL on MSSQL...")
        try:
            df = pd.read_sql(sql, self.sql_engine)
            print(f"  ✅ Query succeeded — {len(df)} rows returned.")
            return {**state, "result": df, "error": None}
        except Exception as e:
            err = str(e)
            print(f"  ✗ Execution error: {err[:300]}")
            return {**state, "result": None, "error": err}

    # ── Conditional edges ─────────────────────────────────────────────────────

    def _route_after_validate(self, state: AgentState) -> str:
        """After validate_sql: go to execute if valid, else back to generate_sql or END."""
        if state["validation_error"]:
            if state["attempts"] < MAX_RETRIES:
                print(f"  → Routing back to generate_sql (validation failed)")
                return "generate_sql"
            else:
                print(f"  → Max retries hit on validation. Ending.")
                return END
        return "execute_sql"

    def _route_after_execute(self, state: AgentState) -> str:
        """After execute_sql: END on success, retry if error and attempts remain."""
        if state["error"] is None:
            return END
        if state["attempts"] < MAX_RETRIES:
            print(f"  → Routing back to generate_sql for correction (attempt {state['attempts']+1})")
            return "generate_sql"
        print(f"  → All {MAX_RETRIES} attempts exhausted. Ending.")
        return END

    # ── Graph builder ─────────────────────────────────────────────────────────

    def _build_graph(self) -> object:
        """Assemble and compile the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("generate_sql", self._node_generate_sql)
        graph.add_node("validate_sql", self._node_validate_sql)
        graph.add_node("execute_sql", self._node_execute_sql)

        # Entry point
        graph.set_entry_point("retrieve")

        # Static edges
        graph.add_edge("retrieve", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")

        # Conditional edges
        graph.add_conditional_edges(
            "validate_sql",
            self._route_after_validate,
            {"generate_sql": "generate_sql", "execute_sql": "execute_sql", END: END},
        )
        graph.add_conditional_edges(
            "execute_sql",
            self._route_after_execute,
            {"generate_sql": "generate_sql", END: END},
        )

        return graph.compile()

    # ── Public API (unchanged interface for api.py) ───────────────────────────

    def generate_sql(self, question: str):
        """
        Run the LangGraph agent.
        Returns: (sql: str, df: DataFrame | None, error: str | None)
        """
        print(f"\n{'='*60}")
        print(f" LangGraph Agent — Query: {question}")
        print(f"{'='*60}")

        initial_state: AgentState = {
            "question": question,
            "tables": [],
            "columns": [],
            "paths": [],
            "prompt": "",
            "sql": "",
            "error": None,
            "result": None,
            "attempts": 0,
            "validation_error": None,
        }

        final_state = self.agent.invoke(initial_state)

        sql = final_state.get("sql", "")
        result = final_state.get("result", None)
        error = final_state.get("error") or final_state.get("validation_error")

        return sql, result, error

    # ── Schema retrieval helper (used by /query/sql-only in api.py) ───────────

    def retrieve_schema_elements(self, query: str):
        """Run only the retrieval step (for /query/sql-only endpoint)."""
        state: AgentState = {
            "question": query,
            "tables": [], "columns": [], "paths": [],
            "prompt": "", "sql": "", "error": None,
            "result": None, "attempts": 0, "validation_error": None,
        }
        state = self._node_retrieve(state)
        return state["tables"], state["columns"]

    def get_join_paths(self, tables: list[str]):
        """Return join paths for a list of tables (for /query/sql-only endpoint)."""
        state: AgentState = {
            "question": "", "tables": tables, "columns": [], "paths": [],
            "prompt": "", "sql": "", "error": None,
            "result": None, "attempts": 0, "validation_error": None,
        }
        # Re-use retrieval node just for the path part
        cypher = """
        MATCH (a:Table {name: $start_node}), (b:Table {name: $end_node})
        MATCH p = shortestPath((a)-[:REFERENCES*..3]-(b))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """
        paths = []
        if len(tables) >= 2:
            with self.graph_driver.session() as session:
                for i in range(len(tables)):
                    for j in range(i + 1, len(tables)):
                        result = session.run(cypher, start_node=tables[i], end_node=tables[j])
                        for record in result:
                            path = record["path_nodes"]
                            if path and len(path) > 1:
                                paths.append(path)
        return paths

    def build_prompt(self, query: str, tables: list[str], columns: list[dict], paths: list[list[str]]):
        """Build the SQL generation prompt (for /query/sql-only endpoint)."""
        return self._build_prompt(query, tables, columns, paths)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str, tables: list[str], columns: list[dict], paths: list[list[str]]) -> str:
        context_schema = "RELEVANT SCHEMA ELEMENTS:\n"
        for tbl in tables:
            context_schema += f"- Table: {tbl}\n"
        context_schema += "\nRELEVANT COLUMNS:\n"
        for col in columns[:TOP_K_COLUMNS]:
            context_schema += f"- {col['table']}.{col['column']} ({col['data_type']})\n"

        context_joins = "\nSUGGESTED JOIN PATHS (Foreign Keys):\n"
        if paths:
            unique_paths = list(set([" -> ".join(p) for p in paths]))
            for p in unique_paths:
                context_joins += f"- {p}\n"
        else:
            context_joins += "- No direct relationship found. Use common sense joins if needed.\n"

        return f"""
You are an expert SQL Generator for a Microsoft SQL Server database (AdventureWorks2019).
Given the user's natural language question, the relevant schema elements, and suggested join paths, generate a valid T-SQL query.

{context_schema}
{context_joins}

USER QUESTION: "{query}"

CONSTRAINTS:
1. Return ONLY the raw SQL code. No markdown formatting (no ```sql), no explanations.
2. Use fully qualified table names (Schema.Table).
3. Use JOINs based on the suggested paths where possible.
4. If a join path is A -> B -> C, use: FROM A JOIN B ON ... JOIN C ON ...
5. Use TOP for limiting results if appropriate.
6. Only generate SELECT statements.

SQL:"""

    def _build_correction_prompt(self, original_prompt: str, failed_sql: str, error: str) -> str:
        return f"""
{original_prompt}

The SQL query you previously generated failed with the following error:

FAILED SQL:
{failed_sql}

ERROR MESSAGE:
{error}

Please analyse the error carefully and generate a corrected T-SQL SELECT query.
Return ONLY the raw SQL. No markdown, no explanation.

CORRECTED SQL:"""
