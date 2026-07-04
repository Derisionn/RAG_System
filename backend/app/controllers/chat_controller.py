import traceback
import uuid
import json
from typing import Generator
from fastapi import HTTPException, BackgroundTasks
from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository
from ..config.config import MAX_RETRIES, PINECONE_HISTORY_NAMESPACE, PINECONE_CHAT_HISTORY_NAMESPACE, SUMMARY_TRIGGER_THRESHOLD

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

    def execute_query(self, question: str, session_id: str, background_tasks: BackgroundTasks, user_id: str) -> dict:
        """Coordinate with RAG service to run query and return formatted results."""
        # 1. Load conversation history from MongoDB
        history = {}
        try:
            history = self.mongo_repo.get_history(user_id, session_id)
        except Exception as e:
            print(f"[ChatController] Warning: could not load history: {e}")

        # 2. Retrieve Semantic History from Pinecone
        try:
            q_vector = self.rag_service.embedding_srv.embed_text(question)
            res = self.rag_service.pinecone_repo.query(
                vector=q_vector,
                top_k=2,
                namespace=PINECONE_HISTORY_NAMESPACE,
                filter={"session_id": session_id}
            )
            semantic_history = []
            for match in res.get("matches", []):
                semantic_history.append({
                    "question": match.get("metadata", {}).get("question", ""),
                    "sql": match.get("metadata", {}).get("sql", "")
                })
            history["semantic_history"] = semantic_history
        except Exception as e:
            print(f"[ChatController] Warning: could not load semantic history: {e}")

        # 3. Retrieve Semantic Chat History from Pinecone (conversational context)
        try:
            res_chat = self.rag_service.pinecone_repo.query(
                vector=q_vector,
                top_k=2,
                namespace=PINECONE_CHAT_HISTORY_NAMESPACE,
                filter={"session_id": session_id}
            )
            semantic_chat_history = []
            for match in res_chat.get("matches", []):
                semantic_chat_history.append({
                    "question": match.get("metadata", {}).get("question", ""),
                    "answer": match.get("metadata", {}).get("answer", "")
                })
            history["semantic_chat_history"] = semantic_chat_history
        except Exception as e:
            print(f"[ChatController] Warning: could not load semantic chat history: {e}")

        try:
            sql, df, error, answer, chart_config = self.rag_service.execute_rag(question, history=history)
        except Exception as exc:
            try:
                self.mongo_repo.save_message(user_id, session_id, question, "", [])
            except:
                pass
            if _is_quota_error(exc):
                raise HTTPException(status_code=429, detail=_QUOTA_DETAIL)
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error:\n{traceback.format_exc()}",
            )

        if error:
            try:
                self.mongo_repo.save_message(user_id, session_id, question, sql or "", [])
            except Exception as e:
                print(f"Error saving failed query to mongo: {e}")
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

        # 4. Schedule background tasks for memory updates
        background_tasks.add_task(self._post_query_tasks, user_id, session_id, question, sql, rows, answer, history)

        return {
            "session_id": session_id,
            "question": question,
            "sql": sql,
            "attempts": self.rag_service.last_attempts,
            "columns": columns,
            "rows": rows,
            "row_count": len(df),
            "answer": answer,
            "chart_config": chart_config,
        }

    def execute_query_stream(self, question: str, session_id: str, background_tasks: BackgroundTasks, user_id: str) -> Generator[str, None, None]:
        """Generator that yields Server-Sent Events (SSE) representing execution progress."""
        history = {}
        try:
            yield f"data: {json.dumps({'step': 'memory', 'message': 'Loading conversation history...'})}\n\n"
            history = self.mongo_repo.get_history(user_id, session_id)
        except Exception as e:
            print(f"[ChatController] Warning: could not load history: {e}")

        try:
            yield f"data: {json.dumps({'step': 'semantic_search', 'message': 'Searching past context...'})}\n\n"
            q_vector = self.rag_service.embedding_srv.embed_text(question)
            res = self.rag_service.pinecone_repo.query(
                vector=q_vector,
                top_k=2,
                namespace=PINECONE_HISTORY_NAMESPACE,
                filter={"session_id": session_id}
            )
            history["semantic_history"] = [{"question": m.get("metadata", {}).get("question", ""), "sql": m.get("metadata", {}).get("sql", "")} for m in res.get("matches", [])]
        except Exception as e:
            print(f"[ChatController] Warning: could not load semantic history: {e}")

        try:
            yield f"data: {json.dumps({'step': 'semantic_search', 'message': 'Searching past chat context...'})}\n\n"
            res_chat = self.rag_service.pinecone_repo.query(
                vector=q_vector,
                top_k=2,
                namespace=PINECONE_CHAT_HISTORY_NAMESPACE,
                filter={"session_id": session_id}
            )
            history["semantic_chat_history"] = [{"question": m.get("metadata", {}).get("question", ""), "answer": m.get("metadata", {}).get("answer", "")} for m in res_chat.get("matches", [])]
        except Exception as e:
            print(f"[ChatController] Warning: could not load semantic chat history: {e}")

        final_state = None
        try:
            for node_name, state in self.rag_service.execute_rag_stream(question, history):
                final_state = state
                if node_name == "planner":
                    plan = state.get("plan", [])
                    step_count = len(plan)
                    yield f"data: {json.dumps({'step': 'planner', 'message': f'Analyzed your request and created a {step_count}-step plan...'})}\n\n"
                elif node_name == "analyze":
                    yield f"data: {json.dumps({'step': 'analyze', 'message': 'Designing multi-step investigation plan...'})}\n\n"
                elif node_name == "dispatcher":
                    step = state.get("current_step", 0)
                    plan = state.get("plan", [])
                    if step < len(plan):
                        action = plan[step].get("action", "")
                        yield f"data: {json.dumps({'step': 'dispatcher', 'message': f'Executing step {step + 1}/{len(plan)}: {action}...'})}\n\n"
                elif node_name == "chat_response":
                    yield f"data: {json.dumps({'step': 'chat_response', 'message': 'Generating conversational response...'})}\n\n"
                elif node_name == "retrieve":
                    yield f"data: {json.dumps({'step': 'retrieve', 'message': 'Retrieved database schema and join paths...'})}\n\n"
                elif node_name == "generate_sql":
                    attempt = state.get("attempts", 1)
                    msg = "Generated SQL query with Gemini..." if attempt <= 1 else f"Self-correcting SQL (attempt {attempt})..."
                    yield f"data: {json.dumps({'step': 'generate_sql', 'message': msg})}\n\n"
                elif node_name == "validate_sql":
                    yield f"data: {json.dumps({'step': 'validate_sql', 'message': 'Validating SQL syntax and safety...'})}\n\n"
                elif node_name == "execute_sql":
                    yield f"data: {json.dumps({'step': 'execute_sql', 'message': 'Executing SQL on Supabase...'})}\n\n"
                elif node_name == "generate_answer":
                    yield f"data: {json.dumps({'step': 'generate_answer', 'message': 'Generating natural language answer...'})}\n\n"
                elif node_name == "generate_chart":
                    yield f"data: {json.dumps({'step': 'generate_chart', 'message': 'Building chart configuration from data...'})}\n\n"
                elif node_name == "synthesize":
                    yield f"data: {json.dumps({'step': 'synthesize', 'message': 'Combining all results into a final response...'})}\n\n"
        except Exception as exc:
            try:
                self.mongo_repo.save_message(user_id, session_id, question, "", [])
            except:
                pass
            if _is_quota_error(exc):
                yield f"data: {json.dumps({'step': 'error', 'errorMsg': _QUOTA_DETAIL['message']})}\n\n"
                return
            yield f"data: {json.dumps({'step': 'error', 'errorMsg': f'Pipeline error: {str(exc)}'})}\n\n"
            return

        error = final_state.get("error") or final_state.get("validation_error")
        sql = final_state.get("sql", "")
        
        if error:
            try:
                self.mongo_repo.save_message(user_id, session_id, question, sql or "", [])
            except:
                pass
            yield f"data: {json.dumps({'step': 'error', 'last_sql': sql, 'errorMsg': f'SQL generation failed: {error[:500]}'})}\n\n"
            return
            
        df = final_state.get("result")
        columns = list(df.columns) if df is not None else []
        rows = df.head(100).to_dict(orient="records") if df is not None else []
        answer = final_state.get("answer")
        chart_config = final_state.get("chart_config")
        
        background_tasks.add_task(self._post_query_tasks, user_id, session_id, question, sql, rows, answer, history)

        yield f"data: {json.dumps({'step': 'complete', 'sql': sql, 'columns': columns, 'rows': rows, 'rowCount': len(df) if df is not None else 0, 'attempts': final_state.get('attempts', 1), 'answer': answer, 'chartConfig': chart_config})}\n\n"

    def _post_query_tasks(self, user_id: str, session_id: str, question: str, sql: str, rows: list[dict], answer: str | None, current_history: dict):
        """Background tasks for saving messages, embedding Q&A pairs, and generating summaries."""
        # 1. ALWAYS save to MongoDB (required for conversational context in recent_messages)
        try:
            self.mongo_repo.save_message(user_id, session_id, question, sql, rows)
        except Exception as e:
            print(f"[ChatController] Error saving message to MongoDB: {e}")

        # 2a. If SQL was generated → embed SQL pair to PINECONE_HISTORY_NAMESPACE
        if sql and sql.strip():
            try:
                vector = self.rag_service.embedding_srv.embed_text(f"Question: {question}\nSQL: {sql}")
                self.rag_service.pinecone_repo.upsert_history(
                    vectors=[{
                        "id": str(uuid.uuid4()),
                        "values": vector,
                        "metadata": {"session_id": session_id, "question": question, "sql": sql}
                    }],
                    namespace=PINECONE_HISTORY_NAMESPACE
                )
            except Exception as e:
                print(f"[ChatController] Error saving SQL pair to Pinecone: {e}")

        # 2b. If this was a conversational answer (no SQL) and it is meaningful enough → embed to chat namespace
        elif answer and len(answer.strip()) > 20:
            try:
                vector = self.rag_service.embedding_srv.embed_text(f"Question: {question}\nAnswer: {answer}")
                self.rag_service.pinecone_repo.upsert_history(
                    vectors=[{
                        "id": str(uuid.uuid4()),
                        "values": vector,
                        "metadata": {"session_id": session_id, "question": question, "answer": answer}
                    }],
                    namespace=PINECONE_CHAT_HISTORY_NAMESPACE
                )
            except Exception as e:
                print(f"[ChatController] Error saving chat pair to Pinecone: {e}")

        # 3. Check for summarization (only for sessions with SQL activity)
        if sql and sql.strip():
            total_msgs = current_history.get("total_messages", 0) + 1
            if total_msgs == 1:
                # Generate initial short title
                try:
                    summary = self.rag_service.reasoner.generate_short_title(question)
                    if summary:
                        self.mongo_repo.save_summary(user_id, session_id, summary)
                except Exception as e:
                    print(f"[ChatController] Error generating short title: {e}")
            elif total_msgs >= SUMMARY_TRIGGER_THRESHOLD and total_msgs % 5 == 0:
                try:
                    all_msgs = self.mongo_repo.get_all_messages(user_id, session_id)
                    summary = self.rag_service.reasoner.summarize_history(all_msgs)
                    if summary:
                        self.mongo_repo.save_summary(user_id, session_id, summary)
                except Exception as e:
                    print(f"[ChatController] Error generating summary: {e}")

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
