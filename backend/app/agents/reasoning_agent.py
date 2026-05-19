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

    def build_prompt(self, question: str, tables: list[str], columns: list[dict], paths: list[list[str]]) -> str:
        """Build standard LLM prompt instructing it to write Supabase compatible PostgreSQL queries."""
        schemas_desc = []
        for tbl in tables:
            tbl_cols = [c for c in columns if c["table_name"] == tbl]
            col_lines = []
            for c in tbl_cols:
                col_lines.append(f"  - {c['column_name']} ({c['data_type']})")
            schemas_desc.append(f"Table: {tbl}\nColumns:\n" + "\n".join(col_lines))

        schema_text = "\n\n".join(schemas_desc)

        paths_text = ""
        if paths:
            paths_desc = []
            for p in paths:
                paths_desc.append(" -> ".join(p))
            paths_text = "Suggested Join Relationships between tables:\n" + "\n".join(paths_desc)

        prompt = f"""You are an expert PostgreSQL DBA and data analyst.
Convert the natural language question into a high-performance PostgreSQL query that is 100% compatible with Supabase database syntax.

Database Schema Context:
{schema_text}

{paths_text}

Rules:
1. Generate standard, ANSI-compliant PostgreSQL queries only.
2. DO NOT use T-SQL or Microsoft SQL Server specific grammar. Use standard PostgreSQL dialect.
3. For limiting results, use 'LIMIT N' at the end of the query. DO NOT use T-SQL 'TOP N'.
4. In PostgreSQL, always double-quote schemas and tables separately as "schema"."table" (e.g., "Sales"."Customer"). NEVER use "schema.table" as a single quoted string, as PostgreSQL will treat it as a single table name with a dot.
5. Ensure all joined tables are linked correctly based on keys.
6. Only return the raw SQL code. DO NOT wrap it in any comments or markup except the query itself.

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
