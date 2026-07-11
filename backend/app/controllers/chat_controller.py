import traceback
import uuid
import json
from typing import Generator
from fastapi import HTTPException, BackgroundTasks
from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository
from ..config.config import MAX_RETRIES, PINECONE_HISTORY_NAMESPACE, PINECONE_CHAT_HISTORY_NAMESPACE, SUMMARY_TRIGGER_THRESHOLD

def _is_quota_error(exc: Exception) -> bool:
    """Return True if exc is a rate-limit error (HTTP 429)."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "rate limit", "too many requests"))

_QUOTA_DETAIL = {
    "message": "Hugging Face API free-tier rate limit exceeded.",
    "action": (
        "The free Hugging Face API is currently experiencing high traffic. "
        "Please wait a few seconds and try again, or consider upgrading to a PRO account."
    ),
}


class ChatController:
    def __init__(self, rag_service: RAGService, mongo_repo: MongoRepository):
        self.rag_service = rag_service
        self.mongo_repo = mongo_repo
        from ..services.cache_service import CacheService
        self.cache = CacheService()

    def execute_query(self, req, question: str, session_id: str, background_tasks: BackgroundTasks, user_id: str) -> dict:
        """Coordinate with RAG service to run query and return formatted results."""
        from ..config.hf_client import request_token_usage
        usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        token = request_token_usage.set(usage_tracker)
        import time
        start_time = time.time()
        # 1. Load conversation history from MongoDB
        history = {}
        t_context = time.time()
        try:
            history = self.mongo_repo.get_history(user_id, session_id)
        except Exception as e:
            print(f"[ChatController] Warning: could not load history: {e}")

        # Check exact question cache
        req_cache_key = self.cache.generate_key("req", question.strip().lower())
        cached_res = self.cache.get(req_cache_key)
        if cached_res:
            print("[ChatController] Exact Question Cache HIT!")
            return {
                "session_id": session_id,
                "question": question,
                "sql": cached_res["sql"],
                "attempts": 1,
                "columns": cached_res["columns"],
                "rows": cached_res["rows"],
                "row_count": len(cached_res["rows"]),
                "answer": cached_res["answer"],
                "chart_config": cached_res.get("chart_config"),
            }

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
        context_ms = (time.time() - t_context) * 1000

        try:
            sql, df, error, answer, chart_config, planner_time_ms, vector_ms, graph_ms, sql_gen_ms, sql_exec_ms, ans_gen_ms = self.rag_service.execute_rag(question, history=history)
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
        rows = json.loads(df.head(100).to_json(orient="records", date_format="iso"))

        # 4. Schedule background tasks for memory updates
        background_tasks.add_task(self._post_query_tasks, user_id, session_id, question, sql, rows, answer, history)

        total_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
        planner_ms = planner_time_ms if 'planner_time_ms' in locals() else 0
        v_ms = vector_ms if 'vector_ms' in locals() else 0
        g_ms = graph_ms if 'graph_ms' in locals() else 0
        c_ms = context_ms if 'context_ms' in locals() else 0
        s_ms = sql_gen_ms if 'sql_gen_ms' in locals() else 0
        e_ms = sql_exec_ms if 'sql_exec_ms' in locals() else 0
        a_ms = ans_gen_ms if 'ans_gen_ms' in locals() else 0
        
        # Standard query has no "handshake" delay, so we just show the route
        route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
        
        cache_hits = []
        if v_ms == 0.0 and total_time_ms > 0: cache_hits.append("Tier 1.5 (Vector)")
        if 0 < g_ms < 15: cache_hits.append("Tier 2 (Graph)")
        if 0 < e_ms < 15: cache_hits.append("Tier 3 (SQL Exec)")
        cache_str = ", ".join(cache_hits) if cache_hits else "None"
        
        box = (
            f"\n=================================================\n"
            f"Request Metrics\n\n"
            f"Connection         : {route_str}\n"
            f"Time to First Byte : N/A (Standard Request)\n"
            f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
            f"Tokens Used        : {usage_tracker['total_tokens']} (P: {usage_tracker['prompt_tokens']}, C: {usage_tracker['completion_tokens']})\n"
            f"Cache Hits         : {cache_str}\n\n"
            f"Planner            : {int(planner_ms)} ms\n"
            f"Context Retriever  : {int(c_ms)} ms\n"
            f"Vector Retriever   : {int(v_ms)} ms\n"
            f"Graph Retriever    : {int(g_ms)} ms\n"
            f"SQL Generation     : {int(s_ms)} ms\n"
            f"SQL Execution      : {int(e_ms)} ms\n"
            f"Answer Generation  : {int(a_ms)} ms\n\n"
            f"================================================="
        )
        import logging
        logging.getLogger("uvicorn").info(box)

        result_dict = {
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
        
        # Save to cache
        if not error:
            self.cache.set(req_cache_key, result_dict, ttl_seconds=3600)
            
        return result_dict

    def execute_query_stream(self, req, question: str, session_id: str, background_tasks: BackgroundTasks, user_id: str):
        from ..config.hf_client import request_token_usage
        import contextvars

        usage_tracker = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ctx = contextvars.copy_context()
        ctx.run(request_token_usage.set, usage_tracker)

        class _ContextIterator:
            def __init__(self, gen, ctx):
                self.gen = gen
                self.ctx = ctx
            def __iter__(self): return self
            def __next__(self): return self.ctx.run(self.gen.__next__)

        gen = self._execute_query_stream_inner(req, question, session_id, background_tasks, user_id, usage_tracker)
        return _ContextIterator(gen, ctx)

    def _execute_query_stream_inner(self, req, question: str, session_id: str, background_tasks: BackgroundTasks, user_id: str, usage_tracker: dict) -> Generator[str, None, None]:
        """Generator that yields Server-Sent Events (SSE) representing execution progress."""
        import time
        import logging
        stream_start_time = time.time()
        logger = logging.getLogger("uvicorn")
        
        t_context = time.time()
        history = {}
        try:
            yield f"data: {json.dumps({'step': 'memory', 'message': 'Loading conversation history...'})}\n\n"
            history = self.mongo_repo.get_history(user_id, session_id)
        except Exception as e:
            print(f"[ChatController] Warning: could not load history: {e}")

        req_cache_key = self.cache.generate_key("req", question.strip().lower())
        cached_res = self.cache.get(req_cache_key)
        if cached_res:
            print("[ChatController] Exact Question Cache HIT for streaming!")
            yield f"data: {json.dumps({'step': 'complete', 'sql': cached_res['sql'], 'columns': cached_res['columns'], 'rows': cached_res['rows'], 'rowCount': cached_res['row_count'], 'attempts': 1, 'answer': cached_res['answer'], 'chartConfig': cached_res.get('chart_config')}, default=str)}\n\n"
            
            # Emit metrics
            total_time_ms = (time.time() - stream_start_time) * 1000
            route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
            box = (
                f"\n=================================================\n"
                f"Request Metrics\n\n"
                f"Connection         : {route_str}\n"
                f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
                f"Cache Hits         : Tier 1 (Exact Question)\n\n"
                f"================================================="
            )
            logger.info(box)
            return

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
        context_ms = (time.time() - t_context) * 1000

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
                
                total_time_ms = (time.time() - stream_start_time) * 1000
                planner_ms = final_state.get('planner_time_ms', 0) if final_state else 0
                v_ms = final_state.get('vector_ms', 0) if final_state else 0
                g_ms = final_state.get('graph_ms', 0) if final_state else 0
                s_ms = final_state.get('sql_gen_ms', 0) if final_state else 0
                e_ms = final_state.get('sql_exec_ms', 0) if final_state else 0
                a_ms = final_state.get('ans_gen_ms', 0) if final_state else 0
                route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
                handshake_ms = getattr(req.state, "handshake_ms", 0.0)
                
                box = (
                    f"\n=================================================\n"
                    f"Request Metrics\n\n"
                    f"Connection         : {route_str}\n"
                    f"Time to First Byte : {handshake_ms:.2f} ms\n"
                    f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
                    f"Tokens Used        : {usage_tracker['total_tokens']} (P: {usage_tracker['prompt_tokens']}, C: {usage_tracker['completion_tokens']})\n\n"
                    f"Planner            : {int(planner_ms)} ms\n"
                    f"Context Retriever  : {int(context_ms)} ms\n"
                    f"Vector Retriever   : {int(v_ms)} ms\n"
                    f"Graph Retriever    : {int(g_ms)} ms\n"
                    f"SQL Generation     : {int(s_ms)} ms\n"
                    f"SQL Execution      : {int(e_ms)} ms\n"
                    f"Answer Generation  : {int(a_ms)} ms\n\n"
                    f"================================================="
                )
                logger.info(box)
                return
                
            yield f"data: {json.dumps({'step': 'error', 'errorMsg': f'Pipeline error: {str(exc)}'})}\n\n"
            total_time_ms = (time.time() - stream_start_time) * 1000
            planner_ms = final_state.get('planner_time_ms', 0) if final_state else 0
            v_ms = final_state.get('vector_ms', 0) if final_state else 0
            g_ms = final_state.get('graph_ms', 0) if final_state else 0
            s_ms = final_state.get('sql_gen_ms', 0) if final_state else 0
            e_ms = final_state.get('sql_exec_ms', 0) if final_state else 0
            a_ms = final_state.get('ans_gen_ms', 0) if final_state else 0
            route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
            handshake_ms = getattr(req.state, "handshake_ms", 0.0)
            box = (
                f"\n=================================================\n"
                f"Request Metrics\n\n"
                f"Connection         : {route_str}\n"
                f"Time to First Byte : {handshake_ms:.2f} ms\n"
                f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
                f"Tokens Used        : {usage_tracker['total_tokens']} (P: {usage_tracker['prompt_tokens']}, C: {usage_tracker['completion_tokens']})\n\n"
                f"Planner            : {int(planner_ms)} ms\n"
                f"Context Retriever  : {int(context_ms)} ms\n"
                f"Vector Retriever   : {int(v_ms)} ms\n"
                f"Graph Retriever    : {int(g_ms)} ms\n"
                f"SQL Generation     : {int(s_ms)} ms\n"
                f"SQL Execution      : {int(e_ms)} ms\n"
                f"Answer Generation  : {int(a_ms)} ms\n\n"
                f"================================================="
            )
            logger.info(box)
            return

        error = final_state.get("error") or final_state.get("validation_error")
        sql = final_state.get("sql", "")
        
        if error:
            try:
                self.mongo_repo.save_message(user_id, session_id, question, sql or "", [])
            except:
                pass
            yield f"data: {json.dumps({'step': 'error', 'last_sql': sql, 'errorMsg': f'SQL generation failed: {error[:500]}'})}\n\n"
            total_time_ms = (time.time() - stream_start_time) * 1000
            planner_ms = final_state.get('planner_time_ms', 0) if final_state else 0
            v_ms = final_state.get('vector_ms', 0) if final_state else 0
            g_ms = final_state.get('graph_ms', 0) if final_state else 0
            s_ms = final_state.get('sql_gen_ms', 0) if final_state else 0
            e_ms = final_state.get('sql_exec_ms', 0) if final_state else 0
            a_ms = final_state.get('ans_gen_ms', 0) if final_state else 0
            route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
            handshake_ms = getattr(req.state, "handshake_ms", 0.0)
            
            cache_hits = []
            if v_ms == 0.0 and total_time_ms > 0: cache_hits.append("Tier 1.5 (Vector)")
            if 0 < g_ms < 15: cache_hits.append("Tier 2 (Graph)")
            if 0 < e_ms < 15: cache_hits.append("Tier 3 (SQL Exec)")
            cache_str = ", ".join(cache_hits) if cache_hits else "None"

            box = (
                f"\n=================================================\n"
                f"Request Metrics\n\n"
                f"Connection         : {route_str}\n"
                f"Time to First Byte : {handshake_ms:.2f} ms\n"
                f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
                f"Tokens Used        : {usage_tracker['total_tokens']} (P: {usage_tracker['prompt_tokens']}, C: {usage_tracker['completion_tokens']})\n"
                f"Cache Hits         : {cache_str}\n\n"
                f"Planner            : {int(planner_ms)} ms\n"
                f"Context Retriever  : {int(context_ms)} ms\n"
                f"Vector Retriever   : {int(v_ms)} ms\n"
                f"Graph Retriever    : {int(g_ms)} ms\n"
                f"SQL Generation     : {int(s_ms)} ms\n"
                f"SQL Execution      : {int(e_ms)} ms\n"
                f"Answer Generation  : {int(a_ms)} ms\n\n"
                f"================================================="
            )
            logger.info(box)
            return
            
        df = final_state.get("result")
        columns = list(df.columns) if df is not None else []
        rows = json.loads(df.head(100).to_json(orient="records", date_format="iso")) if df is not None else []
        answer = final_state.get("answer")
        chart_config = final_state.get("chart_config")
        
        background_tasks.add_task(self._post_query_tasks, user_id, session_id, question, sql, rows, answer, history)

        result_dict = {
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(df) if df is not None else 0,
            "answer": answer,
            "chart_config": chart_config
        }
        
        self.cache.set(req_cache_key, result_dict, ttl_seconds=3600)

        yield f"data: {json.dumps({'step': 'complete', 'sql': sql, 'columns': columns, 'rows': rows, 'rowCount': len(df) if df is not None else 0, 'attempts': final_state.get('attempts', 1), 'answer': answer, 'chartConfig': chart_config}, default=str)}\n\n"
        
        total_time_ms = (time.time() - stream_start_time) * 1000
        planner_ms = final_state.get('planner_time_ms', 0) if final_state else 0
        v_ms = final_state.get('vector_ms', 0) if final_state else 0
        g_ms = final_state.get('graph_ms', 0) if final_state else 0
        s_ms = final_state.get('sql_gen_ms', 0) if final_state else 0
        e_ms = final_state.get('sql_exec_ms', 0) if final_state else 0
        a_ms = final_state.get('ans_gen_ms', 0) if final_state else 0
        route_str = getattr(req.state, "route_str", f"POST {req.url.path}")
        handshake_ms = getattr(req.state, "handshake_ms", 0.0)
        box = (
            f"\n=================================================\n"
            f"Request Metrics\n\n"
            f"Connection         : {route_str}\n"
            f"Time to First Byte : {handshake_ms:.2f} ms\n"
            f"Request Time       : {total_time_ms / 1000.0:.2f} sec\n\n"
            f"Tokens Used        : {usage_tracker['total_tokens']} (P: {usage_tracker['prompt_tokens']}, C: {usage_tracker['completion_tokens']})\n\n"
            f"Planner            : {int(planner_ms)} ms\n"
            f"Context Retriever  : {int(context_ms)} ms\n"
            f"Vector Retriever   : {int(v_ms)} ms\n"
            f"Graph Retriever    : {int(g_ms)} ms\n"
            f"SQL Generation     : {int(s_ms)} ms\n"
            f"SQL Execution      : {int(e_ms)} ms\n"
            f"Answer Generation  : {int(a_ms)} ms\n\n"
            f"================================================="
        )
        logger.info(box)

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
