import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from ..config.config import GEMINI_API_KEY, GEMINI_MODEL

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class ReasoningAgent:
    def __init__(self):
        self.llm = genai.GenerativeModel(GEMINI_MODEL)

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

    def build_prompt(self, question: str, tables: list[str], columns: list[dict], paths: list[list[str]], history: list[dict] | None = None) -> str:
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

        # Build conversation history block
        history_text = ""
        if history:
            history_lines = []
            for msg in history:
                history_lines.append(f"  User: {msg['question']}")
                history_lines.append(f"  SQL:  {msg['sql']}")
            history_text = "Conversation History (for context):\n" + "\n".join(history_lines)

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
