from pydantic import BaseModel, Field
import uuid

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="Who are the top 5 customers by total sales amount?",
        description="Natural language question about the Supabase database.",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation session ID. A new UUID is generated if not provided.",
    )
