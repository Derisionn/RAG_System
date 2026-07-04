from typing import TypedDict, Optional, Any, Generator
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
from ..agents.conversational_agent import ConversationalAgent
from ..agents.intention_agent import IntentionAgent
from ..agents.charting_agent import ChartingAgent
from ..agents.analyzer_agent import AnalyzerAgent
from ..config.config import MAX_RETRIES

# State representation
class AgentState(TypedDict):
    question: str
    plan: list[dict]
    current_step: int
    accumulated_answers: list[str]
    chart_config: Optional[dict]
    history: dict        # conversation memory: summary, semantic_history, messages
    tables: list[str]
    columns: list[dict]
    paths: list[list[str]]
    prompt: str
    sql: str
    error: Optional[str]
    result: Optional[Any]  # Pandas DataFrame
    attempts: int
    validation_error: Optional[str]
    answer: Optional[str]

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
        self.conversational = ConversationalAgent()
        self.intention = IntentionAgent()
        self.charter = ChartingAgent()
        self.analyzer = AnalyzerAgent()

        # 2. Compile LangGraph State Machine
        self.agent = self._build_agent_graph()
        self.last_attempts = 0

    def _build_agent_graph(self):
        workflow = StateGraph(AgentState)

        # Register nodes
        workflow.add_node("planner", self._node_plan)
        workflow.add_node("dispatcher", self._node_dispatcher)
        workflow.add_node("chat_response", self._node_chat_response)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("generate_sql", self._node_generate_sql)
        workflow.add_node("validate_sql", self._node_validate_sql)
        workflow.add_node("execute_sql", self._node_execute_sql)
        workflow.add_node("generate_answer", self._node_generate_answer)
        workflow.add_node("generate_chart", self._node_generate_chart)
        workflow.add_node("analyze", self._node_analyze)
        workflow.add_node("synthesize", self._node_synthesize)

        # Set entry
        workflow.set_entry_point("planner")

        # Edges
        workflow.add_edge("planner", "dispatcher")
        
        workflow.add_conditional_edges(
            "dispatcher",
            self._router_execute_step,
            {
                "chat": "chat_response",
                "sql_query": "retrieve",
                "generate_chart": "generate_chart",
                "analyze": "analyze",
                "synthesize": "synthesize"
            }
        )
        
        # SQL Pipeline edges
        workflow.add_edge("retrieve", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        
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
                "success": "generate_answer",
                "retry": "generate_sql",
                "fail": "dispatcher", # Return to loop on failure
            }
        )

        # Loop back edges (return to dispatcher after step finishes)
        workflow.add_edge("chat_response", "dispatcher")
        workflow.add_edge("generate_answer", "dispatcher")
        workflow.add_edge("generate_chart", "dispatcher")
        workflow.add_edge("analyze", "dispatcher")
        
        # Final exit
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _node_plan(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: plan] Generating execution plan...")
        plan = self.intention.generate_plan(state["question"])
        print(f"  [OK] Plan: {plan}")
        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "accumulated_answers": []
        }

    def _node_dispatcher(self, state: AgentState) -> AgentState:
        """Anchor node for the loop. Also guards against generate_chart with no data."""
        plan = state.get("plan", [])
        step = state.get("current_step", 0)

        if step < len(plan) and plan[step].get("action") == "generate_chart":
            if state.get("result") is None:
                print(f"  [WARNING] generate_chart at step {step} has no data. Auto-injecting sql_query before it.")
                # Insert a sql_query task immediately before the generate_chart step
                new_plan = list(plan)
                new_plan.insert(step, {"action": "sql_query", "parameters": {"query": state["question"]}})
                return {**state, "plan": new_plan}

        return state

    def _node_chat_response(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: chat_response] Generating chat response...")
        # Use the isolated message from task parameters if available,
        # otherwise fall back to the full original question
        step = state.get("current_step", 0)
        plan = state.get("plan", [])
        step_params = plan[step].get("parameters", {}) if step < len(plan) else {}
        task_prompt = step_params.get("message", "").strip() or state["question"]

        answer = self.conversational.generate_chat_response(task_prompt, state.get("history", {}))
        print(f"  [OK] Chat Response: {answer}")
        
        acc = state.get("accumulated_answers", []) + [answer]
        return {
            **state,
            "accumulated_answers": acc,
            "current_step": state["current_step"] + 1
        }

    def _node_retrieve(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: retrieve] Planning schemas...")
        # Build an enriched retrieval query from task parameters
        # so the schema search is as targeted as possible
        step = state.get("current_step", 0)
        plan = state.get("plan", [])
        step_params = plan[step].get("parameters", {}) if step < len(plan) else {}
        parts = [v for v in [
            step_params.get("query", ""),
            step_params.get("metric", ""),
            step_params.get("time", "")
        ] if v]
        retrieval_query = " ".join(parts) if parts else state["question"]
        print(f"  -> Retrieval query: '{retrieval_query}'")

        tables, columns, paths = self.retrieval_srv.retrieve_schema_elements(retrieval_query)
        return {
            **state,
            "tables": tables,
            "columns": columns,
            "paths": paths,
            # ── Reset SQL pipeline state for this fresh step ──
            "attempts": 0,
            "sql": "",
            "prompt": "",
            "error": None,
            "validation_error": None,
            "result": None,
        }

    def _node_generate_sql(self, state: AgentState) -> AgentState:
        attempt = state["attempts"] + 1
        print(f"\n[RAGService Node: generate_sql] Calling Gemini (attempt {attempt})...")

        if state["attempts"] == 0:
            prompt = self.reasoner.build_prompt(
                state["question"], state["tables"], state["columns"], state["paths"],
                history=state.get("history", {})
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
            
            # If execution fails, increment step so loop continues
            acc = state.get("accumulated_answers", []) + [f"Failed to execute SQL: {err_str}"]
            return {
                **state,
                "result": None,
                "error": err_str,
                "accumulated_answers": acc,
                "current_step": state["current_step"] + 1
            }

    def _node_generate_answer(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: generate_answer] Generating natural language answer...")
        answer = self.reasoner.generate_answer(state["question"], state["result"])
        print(f"  [OK] Answer: {answer}")
        
        acc = state.get("accumulated_answers", []) + [answer]
        return {
            **state,
            "answer": answer,
            "accumulated_answers": acc,
            "current_step": state["current_step"] + 1
        }
        
    def _node_generate_chart(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: generate_chart] Generating chart configuration...")
        # Pass chart_type hint from task parameters if the user specified one
        step = state.get("current_step", 0)
        plan = state.get("plan", [])
        step_params = plan[step].get("parameters", {}) if step < len(plan) else {}
        chart_hint = step_params.get("chart_type", "").strip()
        chart_request = f"{state['question']} (chart type: {chart_hint})" if chart_hint else state["question"]

        config = self.charter.generate_chart_config(state.get("result"), chart_request)
        print(f"  [OK] Chart Config Generated")
        
        # We don't add chart raw JSON to accumulated_answers, it's just saved in state
        return {
            **state,
            "chart_config": config,
            "current_step": state["current_step"] + 1
        }

    def _node_analyze(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: analyze] Decomposing investigative question...")
        plan = list(state.get("plan", []))
        step = state.get("current_step", 0)
        
        # Ask AnalyzerAgent to generate an investigation plan
        sub_plan = self.analyzer.generate_investigation_plan(state["question"])
        
        # Replace the current 'analyze' step with the generated sub-steps
        plan = plan[:step] + sub_plan + plan[step+1:]
        
        # We do NOT increment current_step because we want the dispatcher to immediately
        # start executing the first sub-step we just inserted at the current index.
        return {
            **state,
            "plan": plan
        }

    def _node_synthesize(self, state: AgentState) -> AgentState:
        print(f"\n[RAGService Node: synthesize] Synthesizing final response...")
        final_answer = self.conversational.synthesize_answers(state["question"], state.get("accumulated_answers", []))
        return {
            **state,
            "answer": final_answer
        }

    # ── Routers ───────────────────────────────────────────────────────────────

    def _router_execute_step(self, state: AgentState) -> str:
        plan = state.get("plan", [])
        step = state.get("current_step", 0)
        
        if step >= len(plan):
            return "synthesize"
            
        action = plan[step].get("action")
        if action == "chat":
            return "chat"
        elif action == "sql_query":
            return "sql_query"
        elif action == "generate_chart":
            # At this point, dispatcher has already guaranteed result is populated
            return "generate_chart"
        elif action == "analyze":
            return "analyze"

        # fallback
        return "synthesize"

    def _router_after_validate(self, state: AgentState) -> str:
        if state["validation_error"]:
            if state["attempts"] >= MAX_RETRIES:
                print(f"  [WARNING] Validation failed, max attempts reached.")
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

    def execute_rag(self, question: str, history: dict | None = None) -> tuple[str, pd.DataFrame, Optional[str], Optional[str], Optional[dict]]:
        initial_state: AgentState = {
            "question": question,
            "plan": [],
            "current_step": 0,
            "accumulated_answers": [],
            "chart_config": None,
            "history": history or {},
            "tables": [],
            "columns": [],
            "paths": [],
            "prompt": "",
            "sql": "",
            "error": None,
            "result": None,
            "attempts": 0,
            "validation_error": None,
            "answer": None
        }

        final_state = self.agent.invoke(initial_state)
        self.last_attempts = final_state.get("attempts", 0)
        sql = final_state.get("sql", "")
        df = final_state.get("result") if final_state.get("result") is not None else pd.DataFrame()
        error = final_state.get("error") or final_state.get("validation_error")
        answer = final_state.get("answer")
        chart_config = final_state.get("chart_config")

        return sql, df, error, answer, chart_config

    def execute_rag_stream(self, question: str, history: dict | None = None) -> Generator[tuple[str, AgentState], None, None]:
        initial_state: AgentState = {
            "question": question,
            "plan": [],
            "current_step": 0,
            "accumulated_answers": [],
            "chart_config": None,
            "history": history or {},
            "tables": [],
            "columns": [],
            "paths": [],
            "prompt": "",
            "sql": "",
            "error": None,
            "result": None,
            "attempts": 0,
            "validation_error": None,
            "answer": None
        }

        for update in self.agent.stream(initial_state):
            node_name = list(update.keys())[0]
            state = list(update.values())[0]
            self.last_attempts = state.get("attempts", 0)
            yield node_name, state

    def retrieve_schema_elements_only(self, question: str) -> tuple[list[str], list[dict]]:
        return self.retrieval_srv.retrieve_schema_elements(question)[:2]

    def close(self):
        self.postgres_repo.close()
        self.graph_srv.close()
