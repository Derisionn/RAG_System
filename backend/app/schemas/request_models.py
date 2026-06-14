from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="Who are the top 5 customers by total sales amount?",
        description="Natural language question about the Supabase database.",
    )
