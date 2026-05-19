"""
rag_pipeline.py
───────────────
Agentic Hybrid SQL RAG System powered by LangGraph.

Graph nodes:
  retrieve      → Semantic (Pinecone) + Relational (Neo4j) retrieval
  generate_sql  → Gemini LLM generates or corrects SQL
  validate_sql  → Safety + syntax pre-check (SELECT-only, sqlparse)
  execute_sql   → Run SQL on Supabase via SQLAlchemy

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

import requests
from sqlalchemy import create_engine
from pinecone import Pinecone
from neo4j import GraphDatabase
import google.generativeai as genai

from langgraph.graph import StateGraph, END

from .config import (
    EMBEDDING_MODEL,
    HF_API_TOKEN,
    HF_API_URL,
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

        # 1. Pinecone  (no local embedding model — Gemini API used at query time)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vector_index = self.pc.Index(PINECONE_INDEX)

        # 2. Neo4j — short connection lifetime prevents stale sessions on Render
        self.graph_driver = self._make_neo4j_driver()

        # 3. Gemini  (single client used for both embeddings + generation)
        self.llm = genai.GenerativeModel(GEMINI_MODEL)

        # 4. SQL engine
        self.sql_engine = create_engine(CONNECTION_STRING)

        # 5. Build and compile the LangGraph agent
        self.agent = self._build_graph()
        self.last_attempts = 0  # tracks actual attempts used in last generate_sql call
        print("LangGraph agent compiled ✅")

    def _make_neo4j_driver(self):
        """Create a Neo4j driver with pool settings that avoid stale connections."""
        return GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PWD),
            max_connection_lifetime=200,      # recycle before AuraDB closes them (~300s)
            max_connection_pool_size=5,
            connection_acquisition_timeout=30,
        )

    def close(self):
        self.graph_driver.close()

    # ── Embedding helper ───────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """
        Embed a single string via HuggingFace Inference API.
        Uses the same all-MiniLM-L6-v2 model as the Pinecone index — no local torch.
        """
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # HF feature-extraction returns: list[list[float]] (one vector per input)
        # For a single string input it returns a single list[float]
        if isinstance(data[0], list):
            return data[0]  # nested: [[0.1, 0.2, ...]]
        return data           # flat: [0.1, 0.2, ...]

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        """Call Gemini LLM with exponential backoff on ResourceExhausted (429) rate limit."""
        import time
        from google.api_core.exceptions import ResourceExhausted
        
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"  ⚠️ Gemini Rate Limit Exceeded (429). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise e

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """
        Node 1 — Retrieve
        Semantic search via Pinecone + join-path search via Neo4j.
        """
        question = state["question"]
        print(f"\n[Node: retrieve] Embedding query via HF API: '{question}'")

        # Semantic retrieval — embed via HuggingFace Inference API (no local model)
        query_vec = self._embed(question)
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
            try:
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
            except Exception as e:
                print(f"  ⚠️ Neo4j session error ({type(e).__name__}): {e}. Recreating driver and retrying...")
                try:
                    self.graph_driver.close()
                except Exception:
                    pass
                try:
                    self.graph_driver = self._make_neo4j_driver()
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
                except Exception as inner_e:
                    print(f"  ❌ Graceful degradation: Neo4j retry also failed: {inner_e}. Proceeding without join paths.")

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

        response_text = self._call_llm_with_retry(prompt)
        sql = response_text.strip().replace("```sql", "").replace("```", "").strip()
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
        Runs the validated SQL on Supabase and returns a DataFrame.
        """
        sql = state["sql"]
        print(f"\n[Node: execute_sql] Running SQL on Supabase...")
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
        self.last_attempts = final_state.get("attempts", 0)  # store for api.py

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

    def _find_join_paths(self, tables: list[str]) -> list[list[str]]:
        """Shared helper — find shortest join paths between all table pairs via Neo4j."""
        cypher = """
        MATCH (a:Table {name: $start_node}), (b:Table {name: $end_node})
        MATCH p = shortestPath((a)-[:REFERENCES*..3]-(b))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """
        paths: list[list[str]] = []
        if len(tables) >= 2:
            try:
                with self.graph_driver.session() as session:
                    for i in range(len(tables)):
                        for j in range(i + 1, len(tables)):
                            result = session.run(cypher, start_node=tables[i], end_node=tables[j])
                            for record in result:
                                path = record["path_nodes"]
                                if path and len(path) > 1:
                                    paths.append(path)
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to retrieve join paths from Neo4j in _find_join_paths: {e}. Proceeding without paths.")
        return paths

    def get_join_paths(self, tables: list[str]) -> list[list[str]]:
        """Return join paths for a list of tables (for /query/sql-only endpoint)."""
        return self._find_join_paths(tables)

    def build_prompt(self, query: str, tables: list[str], columns: list[dict], paths: list[list[str]]):
        """Build the SQL generation prompt (for /query/sql-only endpoint)."""
        return self._build_prompt(query, tables, columns, paths)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str, tables: list[str], columns: list[dict], paths: list[list[str]]) -> str:
        context_schema = "RELEVANT SCHEMA ELEMENTS:\n"
        for tbl in tables:
            if "." in tbl:
                schema_name, table_name = tbl.split(".", 1)
                quoted_tbl = f'"{schema_name}"."{table_name}"'
            else:
                quoted_tbl = f'"{tbl}"'
            context_schema += f"- Table: {quoted_tbl}\n"
            
        context_schema += "\nRELEVANT COLUMNS:\n"
        for col in columns[:TOP_K_COLUMNS]:
            tbl = col['table']
            if "." in tbl:
                schema_name, table_name = tbl.split(".", 1)
                quoted_tbl = f'"{schema_name}"."{table_name}"'
            else:
                quoted_tbl = f'"{tbl}"'
            context_schema += f"- {quoted_tbl}.\"{col['column']}\" ({col['data_type']})\n"

        context_joins = "\nSUGGESTED JOIN PATHS (Foreign Keys):\n"
        if paths:
            quoted_paths = []
            for p in paths:
                quoted_path_parts = []
                for node in p:
                    if "." in node:
                        s_name, t_name = node.split(".", 1)
                        quoted_path_parts.append(f'"{s_name}"."{t_name}"')
                    else:
                        quoted_path_parts.append(f'"{node}"')
                quoted_paths.append(" -> ".join(quoted_path_parts))
            unique_paths = list(set(quoted_paths))
            for p in unique_paths:
                context_joins += f"- {p}\n"
        else:
            context_joins += "- No direct relationship found. Use common sense joins if needed.\n"

        return f"""
You are an expert SQL Generator for a Supabase (PostgreSQL) database.
Given the user's natural language question, the relevant schema elements, and suggested join paths, generate a valid PostgreSQL query.

{context_schema}
{context_joins}

USER QUESTION: "{query}"

CONSTRAINTS:
1. Return ONLY the raw SQL code. No markdown formatting (no ```sql), no explanations.
2. In PostgreSQL, always double-quote schemas and tables separately as "schema"."table" (e.g., "public"."customer"). NEVER use "schema.table" as a single quoted string, as PostgreSQL will treat it as a single table name with a dot.
3. Use JOINs based on the suggested paths where possible.
4. If a join path is A -> B -> C, use: FROM A JOIN B ON ... JOIN C ON ...
5. Use LIMIT for limiting results if appropriate (DO NOT use TOP).
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

Please analyse the error carefully and generate a corrected PostgreSQL SELECT query.
Return ONLY the raw SQL. No markdown, no explanation.

CORRECTED SQL:"""
