from typing import TypedDict, Optional, Any
import pandas as pd
from langgraph.graph import StateGraph, END

from ..repositories.pinecone_repository import PineconeRepository
from ..repositories.postgres_repository import PostgresRepository
from .embedding_service import EmbeddingService
from .graph_service import GraphService
from .retrieval_service import RetrievalService
from ..agents.reasoning_agent import ReasoningAgent
from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.planner_agent import PlannerAgent
from ..config.config import MAX_RETRIES

# State representation
class AgentState(TypedDict):
    question: str
    history: list[dict]        # conversation history from MongoDB
    tables: list[str]
    columns: list[dict]
    paths: list[list[str]]
    prompt: str
    sql: str
    error: Optional[str]
    result: Optional[Any]  # Pandas DataFrame
    attempts: int
    validation_error: Optional[str]

class RAGService:
    def __init__(self):
        # 1. Initialize repos & services
        self.pinecone_repo = PineconeRepository()
        self.postgres_repo = PostgresRepository()
        self.embedding_srv = EmbeddingService()
        self.graph_srv = GraphService()
        self.planner = PlannerAgent()

        self.retrieval_srv = RetrievalService(
            self.pinecone_repo, self.embedding_srv, self.graph_srv, self.planner
        )
        self.reasoner = ReasoningAgent()
        self.evaluator = EvaluatorAgent()

        # 2. Compile LangGraph State Machine
        self.agent = self._build_agent_graph()
        self.last_attempts = 0

    def _build_agent_graph(self):
        workflow = StateGraph(AgentState)

        # Register nodes
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("generate_sql", self._node_generate_sql)
        workflow.add_node("validate_sql", self._node_validate_sql)
        workflow.add_node("execute_sql", self._node_execute_sql)

        # Set entry
        workflow.set_entry_point("retrieve")

        # Routing edges
        workflow.add_conditional_edges(
            "validate_sql",
            self._router_after_validate,
            {
                "execute": "execute_sql",
                "correct": "generate_sql",
            }
        )
        workflow.add_conditional_edges(
            "execute_sql",
            self._router_after_execute,
            {
                "success": END,
                "retry": "generate_sql",
                "fail": END,
            }
        )

        # Set linear edges
        workflow.add_edge("retrieve", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")

        return workflow.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """Retrieve node to plan relevant schemas and join paths."""
        tables, columns, paths = self.retrieval_srv.retrieve_schema_elements(state["question"])
        return {
            **state,
            "tables": tables,
            "columns": columns,
            "paths": paths
        }

    def _node_generate_sql(self, state: AgentState) -> AgentState:
        """Call Gemini model to generate or self-correct the query."""
        attempt = state["attempts"] + 1
        print(f"\n[RAGService Node: generate_sql] Calling Gemini (attempt {attempt})...")

        if state["attempts"] == 0:
            prompt = self.reasoner.build_prompt(
                state["question"], state["tables"], state["columns"], state["paths"],
                history=state.get("history", [])
            )
        else:
            prompt = self.reasoner.build_correction_prompt(
                state["prompt"], state["sql"], state["error"] or state["validation_error"] or ""
            )

        sql = self.reasoner.generate_sql(prompt)
        print(f"  -> SQL generated:\n{sql[:300]}")

        return {
            **state,
            "prompt": prompt,
            "sql": sql,
            "attempts": attempt,
            "error": None,
            "validation_error": None,
        }


    def _node_validate_sql(self, state: AgentState) -> AgentState:
        """Validate safety and syntax structure of the generated query."""
        print(f"\n[RAGService Node: validate_sql] Validating generated SQL...")
        val_error = self.evaluator.validate_sql(state["sql"])
        if val_error:
            print(f"  [ERROR] Validation failed: {val_error}")
        else:
            print("  [OK] Validation passed.")
        
        return {
            **state,
            "validation_error": val_error
        }

    def _node_execute_sql(self, state: AgentState) -> AgentState:
        """Execute the SQL query on Supabase database."""
        print(f"\n[RAGService Node: execute_sql] Running query on Supabase...")
        try:
            df = self.postgres_repo.execute_query(state["sql"])
            print(f"  [OK] Execution successful ({len(df)} rows).")
            return {
                **state,
                "result": df,
                "error": None
            }
        except Exception as e:
            err_str = str(e)
            print(f"  [ERROR] Execution failed: {err_str[:200]}")
            return {
                **state,
                "result": None,
                "error": err_str
            }

    # ── Routers ───────────────────────────────────────────────────────────────

    def _router_after_validate(self, state: AgentState) -> str:
        if state["validation_error"]:
            if state["attempts"] >= MAX_RETRIES:
                print(f"  [WARNING] Validation failed, but max attempts ({MAX_RETRIES}) reached. Forcing execution.")
                return "execute"
            return "correct"
        return "execute"

    def _router_after_execute(self, state: AgentState) -> str:
        if state["error"]:
            if state["attempts"] >= MAX_RETRIES:
                print(f"  [WARNING] Execution failed after {state['attempts']} attempts. Aborting.")
                return "fail"
            return "retry"
        return "success"

    # ── Pipeline Interface ────────────────────────────────────────────────────

    def execute_rag(self, question: str, history: list[dict] | None = None) -> tuple[str, pd.DataFrame, Optional[str]]:
        """
        Execute full RAG pipeline returning (sql, df, error).
        Accepts optional conversation history for context-aware SQL generation.
        """
        initial_state: AgentState = {
            "question": question,
            "history": history or [],
            "tables": [],
            "columns": [],
            "paths": [],
            "prompt": "",
            "sql": "",
            "error": None,
            "result": None,
            "attempts": 0,
            "validation_error": None
        }

        final_state = self.agent.invoke(initial_state)
        
        self.last_attempts = final_state["attempts"]
        sql = final_state["sql"]
        df = final_state["result"] if final_state["result"] is not None else pd.DataFrame()
        error = final_state["error"] or final_state["validation_error"]

        return sql, df, error

    def retrieve_schema_elements_only(self, question: str) -> tuple[list[str], list[dict]]:
        return self.retrieval_srv.retrieve_schema_elements(question)[:2]

    def close(self):
        self.postgres_repo.close()
        self.graph_srv.close()
