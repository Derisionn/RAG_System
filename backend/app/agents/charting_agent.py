import time
import json
import pandas as pd
from ..config.hf_client import hf_model

class ChartingAgent:
    def __init__(self):
        self.llm = hf_model

    def generate_chart_config(self, chart_type: str, data: list[dict], max_retries: int = 3) -> dict:
        prompt = f"""You are a Frontend Data Visualization Expert.
Your task is to generate a JSON configuration for Recharts to visualize the provided data.
The requested chart type is: {chart_type}

Data (up to 50 rows):
{json.dumps(data[:50])}

Output STRICTLY JSON with the following structure:
{{
  "type": "bar" | "line" | "pie",
  "xAxisKey": "string (the primary category column name)",
  "yAxisKey": "string (the numerical value column)",
  "colors": ["#hex1", "#hex2"]
}}
"""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response_text = self.llm.invoke(prompt)
                
                import re
                text = response_text.strip()
                match = re.search(r"(\{.*\})", text.replace('\n', ' '), re.DOTALL)
                if match:
                    text = match.group(1).strip()
                
                config = json.loads(text)
                return config
            except Exception as e:
                print(f"[ChartingAgent] Error generating config (attempt {attempt + 1}): {e}")
                time.sleep(delay)
                delay *= 2

        # Fallback config
        if not data: return {}
        keys = list(data[0].keys())
        return {
            "type": chart_type if chart_type in ["bar", "line", "pie"] else "bar",
            "xAxisKey": keys[0],
            "yAxisKey": keys[1] if len(keys) > 1 else keys[0],
            "colors": ["#8884d8", "#82ca9d"]
        }
