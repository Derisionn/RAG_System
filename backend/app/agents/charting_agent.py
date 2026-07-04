import time
import json
import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from ..config.gemini_client import gemini_model

class ChartingAgent:
    def __init__(self):
        self.llm = gemini_model

    def generate_chart_config(self, df: pd.DataFrame, request: str, max_retries: int = 3) -> dict:
        """
        Takes raw data and generates a JSON configuration for a frontend charting library.
        Returns a dictionary with 'type' and 'data' (or whatever format the frontend needs).
        """
        if df is None or df.empty:
            return {"error": "No data available to chart."}
            
        # Sample the data to give the LLM context of columns and values
        data_subset = df.head(5).to_dict(orient="records")
        columns = list(df.columns)
        
        prompt = f"""You are a data visualization expert.
The user wants to chart the following data based on this request: "{request}"

Here are the columns: {columns}
Here is a sample of the data: {data_subset}

Determine the best chart type (e.g., 'bar', 'line', 'pie') and map the data columns to appropriate axes.
Return ONLY a valid JSON object with the following structure:
{{
  "chartType": "bar",
  "xAxisKey": "column_name_for_x",
  "yAxisKey": "column_name_for_y",
  "description": "A brief sentence explaining the chart"
}}

Do not include markdown tags like ```json."""

        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                
                config = json.loads(text.strip())
                # Attach the actual raw data to the config so the frontend has it
                config["data"] = df.to_dict(orient="records")
                return config
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"[ChartingAgent] Error generating chart config: {e}")
                return {"error": "Failed to generate chart configuration."}
