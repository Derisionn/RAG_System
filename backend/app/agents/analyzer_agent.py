import json
import time
from ..config.hf_client import hf_structured_model, hf_model

class AnalyzerAgent:
    """
    Agent responsible for breaking down complex analytical 'investigation' questions
    into a sequence of SQL queries and charts.
    """

    def __init__(self):
        # Use the structured model so the output is guaranteed to be valid JSON
        self.llm = hf_structured_model
        self.llm_plain = hf_model

    def generate_investigation_plan(self, question: str, max_retries: int = 3) -> list[dict]:
        """
        Decomposes a 'Why / What caused / How did' question into an ordered
        sequence of investigative tasks.
        Each task is one of: 'sql_query' | 'generate_chart'.
        Returns a list of task dicts ready to be injected into the plan.
        """
        prompt = f"""You are a Senior Data Analyst AI.
The user has asked an investigative or causal question that cannot be answered with a single query.
Your job is to design a systematic, data-driven investigation plan.

The plan should:
1. Start by confirming the problem with a high-level comparison query (e.g., this period vs. last period).
2. Break the problem down by relevant dimensions (e.g., by category, by region, by product, by sales rep).
3. Add a 'generate_chart' step after each important sql_query to visualize the findings.
4. Keep the plan focused — aim for 3 to 5 steps maximum.

Available actions:
- "sql_query": Run a SQL query on the database. Use 'query' in parameters to describe what to fetch.
- "generate_chart": Visualize the data from the previous sql_query. Use 'chart_type' (bar, line, pie) in parameters.

Do NOT include a 'chat' step. Focus purely on data investigation.

User Question: {question}

Return a plan with tasks that will systematically investigate the root cause.
Output STRICTLY JSON exactly matching this format:
{{"tasks": [{{"action": "sql_query", "parameters": {{"query": "..."}}}}]}}
"""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response_text = self.llm.invoke(prompt)
                plan_data = json.loads(response_text)
                tasks = plan_data.get("tasks", [])

                # Filter to only valid investigative actions (no chat)
                valid_actions = {"sql_query", "generate_chart"}
                filtered = [t for t in tasks if t.get("action") in valid_actions]

                if not filtered:
                    # Fallback: simple two-step plan
                    return self._fallback_plan(question)

                print(f"  [AnalyzerAgent] Generated {len(filtered)}-step investigation plan.")
                return filtered

            except Exception as e:
                print(f"[AnalyzerAgent] Error generating plan (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return self._fallback_plan(question)
                time.sleep(delay)
                delay *= 2

    def _fallback_plan(self, question: str) -> list[dict]:
        """Minimal 2-step fallback when structured generation fails."""
        return [
            {"action": "sql_query", "parameters": {"query": question}},
            {"action": "generate_chart", "parameters": {"chart_type": "bar"}},
        ]
