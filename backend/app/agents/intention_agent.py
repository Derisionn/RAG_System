import time
import json
from google.api_core.exceptions import ResourceExhausted
from ..config.gemini_client import gemini_structured_model, gemini_model, Plan


class IntentionAgent:
    def __init__(self):
        # Structured output model — forces Gemini to return valid JSON matching Plan schema
        self.llm = gemini_structured_model
        # Plain fallback model
        self.llm_plain = gemini_model

    def generate_plan(self, question: str, max_retries: int = 3) -> list[dict]:
        """
        Parses the user's question and returns a list of task dicts.
        Uses Gemini Structured Outputs to guarantee valid JSON matching the Plan schema.
        Available actions: 'chat', 'sql_query', 'generate_chart'.
        """
        prompt = f"""You are the Master Orchestrator for a Database AI.
Your job is to break the user's request into a series of actionable steps.

You have access to the following actions:
1. "chat" - For greetings, pleasantries, or questions that don't need data.
2. "sql_query" - For retrieving data, metrics, or answering analytical questions from a database.
3. "generate_chart" - For visualizing data. ALWAYS precede this with a 'sql_query' task to get the data first.
4. "analyze" - For investigative or causal questions like "Why are sales down?" or "What caused the spike?". This action will automatically generate a multi-step investigation plan. Use 'topic' in parameters.

Each task must have an 'action' and a 'parameters' object.

Example 1 (multi-step):
User: "Hi, what was the revenue last month?"
→ tasks: [chat (message="Hi"), sql_query (metric="revenue", time="last month")]

Example 2 (investigation):
User: "Why are our electronics sales declining?"
→ tasks: [analyze (topic="electronics sales decline")]

Example 2 (chart):
User: "Plot a bar chart of the top 5 customers."
→ tasks: [sql_query (query="top 5 customers by revenue"), generate_chart (chart_type="bar")]

User Request: {question}"""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                # With response_mime_type=application/json, response.text is guaranteed clean JSON
                plan_data = json.loads(response.text)
                tasks = plan_data.get("tasks", [])
                if not tasks:
                    return [{"action": "sql_query", "parameters": {}}]
                return tasks
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"  [WARNING] IntentionAgent rate limited. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"[IntentionAgent] Error generating plan (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return [{"action": "sql_query", "parameters": {"query": question}}]
                time.sleep(delay)
                delay *= 2
