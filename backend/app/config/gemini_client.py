"""
gemini_client.py — Shared Gemini model instances for all agents.

Centralising model creation here means:
- Switching models (e.g., gemini-pro) only requires changing one file.
- Adding system instructions or safety settings applies everywhere at once.
- No duplicate client objects wasting memory across agents.
"""
import google.generativeai as genai
from pydantic import BaseModel
from .config import GEMINI_API_KEY, GEMINI_MODEL

genai.configure(api_key=GEMINI_API_KEY)


from typing import Optional

# ── Pydantic schema for IntentionAgent structured output ─────────────────────
# Defined here to avoid circular import (config → agents → config).

class TaskParameters(BaseModel):
    metric: Optional[str]
    time: Optional[str]
    query: Optional[str]
    chart_type: Optional[str]
    message: Optional[str]
    topic: Optional[str]


class Task(BaseModel):
    action: str   # "chat" | "sql_query" | "generate_chart" | "analyze"
    parameters: TaskParameters


class Plan(BaseModel):
    tasks: list[Task]


# ── Standard model (for reasoning, conversational, charting) ──────────────────
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# ── Structured output model (for IntentionAgent) ──────────────────────────────
# Forces Gemini to return valid JSON matching the Plan schema — no string parsing needed.
gemini_structured_model = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=Plan,
    )
)
