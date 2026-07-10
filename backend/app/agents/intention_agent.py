import time
import json
from ..config.hf_client import hf_structured_model, hf_model


class IntentionAgent:
    def __init__(self):
        # Structured output model — attempts to force JSON
        self.llm = hf_structured_model
        # Plain fallback model
        self.llm_plain = hf_model

    def generate_plan(self, question: str, history: dict | None = None, max_retries: int = 3) -> list[dict]:
        """
        Parses the user's question and returns a list of task dicts.
        Uses Hugging Face wrapped in a JSON prompt.
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
→ {{"tasks": [{{"action": "chat", "parameters": {{"message": "Hi"}}}}, {{"action": "sql_query", "parameters": {{"metric": "revenue", "time": "last month"}}}}]}}

Example 2 (investigation):
User: "Why are our electronics sales declining?"
→ {{"tasks": [{{"action": "analyze", "parameters": {{"topic": "electronics sales decline"}}}}]}}

Example 3 (chart):
User: "Plot a bar chart of the top 5 customers."
→ {{"tasks": [{{"action": "sql_query", "parameters": {{"query": "top 5 customers by revenue"}}}}, {{"action": "generate_chart", "parameters": {{"chart_type": "bar"}}}}]}}
"""

        history_text = ""
        if history and history.get("messages"):
            recent_lines = ["\nRecent Conversation History:"]
            # Get last 3 interactions to provide context for pronouns
            for msg in history["messages"][-3:]:
                recent_lines.append(f"User: {msg['question']}\nAssistant: {msg.get('answer', '...')}")
            history_text = "\n".join(recent_lines) + "\n"

        prompt += f"{history_text}\nUser Request: {question}"

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response_text = self.llm.invoke(prompt)
                plan_data = json.loads(response_text)
                tasks = plan_data.get("tasks", [])
                if not tasks:
                    return [{"action": "sql_query", "parameters": {}}]
                return tasks
            except Exception as e:
                print(f"[IntentionAgent] Error generating plan (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return [{"action": "sql_query", "parameters": {"query": question}}]
                time.sleep(delay)
                delay *= 2
