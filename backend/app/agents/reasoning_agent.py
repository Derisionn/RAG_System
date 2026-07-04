import time
import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from ..config.gemini_client import gemini_model

class ReasoningAgent:
    def __init__(self):
        self.llm = gemini_model

    def generate_sql(self, prompt: str, max_retries: int = 5) -> str:
        """Call Gemini LLM with exponential backoff on ResourceExhausted (429) rate limit."""
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text.strip().replace("```sql", "").replace("```", "").strip()
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"  [WARNING] Gemini Rate Limit Exceeded (429). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise e

    def summarize_history(self, messages: list[dict], max_retries: int = 3) -> str:
        """Ask Gemini to summarize the older conversation history."""
        if not messages:
            return ""
        
        history_lines = []
        for msg in messages:
            history_lines.append(f"User: {msg['question']}\nSQL: {msg['sql']}")
            
        history_text = "\n\n".join(history_lines)
        prompt = f"""Summarize the following SQL conversation history into a concise paragraph.
Focus on the tables they explored, the filters they applied, and the general intent.

Conversation:
{history_text}

Summary:"""
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text.strip()
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                raise e

    def generate_short_title(self, question: str, max_retries: int = 2) -> str:
        """Generate a short 3-5 word title for a new session based on the first question."""
        prompt = f"Summarize the following question into a short title of 3-6 words. Do not use quotes or special formatting.\n\nQuestion: {question}\n\nTitle:"
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text.strip().replace('"', '')
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                return ""

    def build_prompt(self, question: str, tables: list[str], columns: list[dict], paths: list[list[str]], history: dict | None = None) -> str:
        """Build standard LLM prompt instructing it to write Supabase compatible PostgreSQL queries."""
        schemas_desc = []
        for tbl in tables:
            tbl_lower = tbl.lower()
            tbl_cols = [c for c in columns if c["table_name"].lower() == tbl_lower]
            col_lines = []
            for c in tbl_cols:
                col_lines.append(f"  - {c['column_name'].lower()} ({c['data_type']})")
            
            if "." in tbl_lower:
                schema_name, table_name = tbl_lower.split(".", 1)
                quoted_tbl = f"{schema_name}.{table_name}"
            else:
                quoted_tbl = tbl_lower
            
            schemas_desc.append(f"Table: {quoted_tbl}\nColumns:\n" + "\n".join(col_lines))

        schema_text = "\n\n".join(schemas_desc)

        paths_text = ""
        if paths:
            paths_desc = []
            for p in paths:
                quoted_path_parts = []
                for node in p:
                    if "." in node:
                        s_name, t_name = node.split(".", 1)
                        quoted_path_parts.append(f"{s_name}.{t_name}")
                    else:
                        quoted_path_parts.append(node)
                paths_desc.append(" -> ".join(quoted_path_parts))
            paths_text = "Suggested Join Relationships between tables:\n" + "\n".join(paths_desc)

        # Build 3-layer memory block
        history_text = ""
        if history:
            blocks = []
            if history.get("summary"):
                blocks.append("─── CONVERSATION SUMMARY (from older messages) ───\n" + history["summary"])
                
            if history.get("semantic_history"):
                semantic_lines = ["─── RELEVANT PAST QUERIES ───"]
                for msg in history["semantic_history"]:
                    semantic_lines.append(f"  User: {msg['question']}\n  SQL:  {msg['sql']}")
                blocks.append("\n".join(semantic_lines))
                
            if history.get("messages"):
                recent_lines = ["─── RECENT HISTORY (immediate context) ───"]
                for msg in history["messages"]:
                    recent_lines.append(f"  User: {msg['question']}\n  SQL:  {msg['sql']}")
                blocks.append("\n".join(recent_lines))
                
            if blocks:
                history_text = "Conversation Memory:\n" + "\n\n".join(blocks)

        prompt = f"""You are an expert PostgreSQL DBA and data analyst.
Convert the natural language question into a high-performance PostgreSQL query that is 100% compatible with Supabase database syntax.

Database Schema Context:
{schema_text}

{paths_text}

{history_text}

Rules:
1. Generate standard, ANSI-compliant PostgreSQL queries only.
2. DO NOT use T-SQL or Microsoft SQL Server specific grammar. Use standard PostgreSQL dialect.
3. For limiting results, use 'LIMIT N' at the end of the query. DO NOT use T-SQL 'TOP N'.
4. CRITICAL: PostgreSQL stores all identifiers in lowercase unless they were created with double-quotes. NEVER use double-quotes around schema names, table names, or column names. Always write them in lowercase without quotes (e.g., sales.salesorderheader, not "sales"."SalesOrderHeader"). Using quoted mixed-case identifiers WILL cause "relation does not exist" errors.
5. Ensure all joined tables are linked correctly based on keys.
6. Only return the raw SQL code. DO NOT wrap it in any comments or markup except the query itself.
7. If the question refers to results from a previous query (e.g. "their", "those", "the same"), use the Conversation History above to understand the context.
8. All schema names, table names, and column names must be fully lowercase and unquoted.

Question: {question}
SQL:"""
        return prompt

    def build_correction_prompt(self, original_prompt: str, failed_sql: str, error_msg: str) -> str:
        """Build prompt for self-correction when a query fails execution."""
        prompt = f"""{original_prompt}

The previous query you generated failed execution:
```sql
{failed_sql}
```

Database Execution Error:
{error_msg}

Please analyze this database execution error and generate a corrected PostgreSQL query that fixes the issue. Return ONLY the raw SQL code.
Corrected SQL:"""
        return prompt

    def generate_answer(self, question: str, df: pd.DataFrame, max_retries: int = 3) -> str:
        """Generate a natural language answer based on the user's question and the SQL results."""
        if df is None or df.empty:
            return "No data was returned for this query."
            
        # Only take top 5 rows to save tokens and latency
        data_subset = df.head(5).to_dict(orient="records")
        row_count = len(df)
        
        prompt = f"""You are a helpful AI data assistant.
A user asked a question about the AdventureWorks database. We ran a SQL query to get the answer.
Please provide a clear, concise, and natural-sounding sentence that answers the user's question using the provided data.
DO NOT explain how you got the data, DO NOT show the SQL, just answer the question directly as if you were talking to them.
If there are many rows, just summarize the top results or mention the total count ({row_count} total rows).

Question: {question}

Data (Top 5 rows):
{data_subset}

Natural Language Answer:"""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                return response.text.strip()
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"[ReasoningAgent] Error generating answer: {e}")
                return "Here are the results for your query."
