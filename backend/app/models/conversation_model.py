from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any

class ChatMessage(BaseModel):
    """Represents a single Q&A turn in a conversation."""
    question: str
    sql: str
    row_preview: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime

class ConversationSession(BaseModel):
    """Represents an entire conversation session stored in MongoDB."""
    session_id: str
    user_id: str
    created_at: datetime
    summary: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
