from fastapi import APIRouter, Depends
from ..models.request_models import QueryRequest
from ..models.response_models import QueryResponse, SQLOnlyResponse
from ..controllers.chat_controller import ChatController
from ..services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["RAG"])

# Shared single RAGService instance
_rag_service: RAGService | None = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_chat_controller(rag_service: RAGService = Depends(get_rag_service)) -> ChatController:
    return ChatController(rag_service)

@router.post("", response_model=QueryResponse)
def execute_rag_query(request: QueryRequest, controller: ChatController = Depends(get_chat_controller)):
    """Full RAG query execution against Supabase PostgreSQL database."""
    res = controller.execute_query(request.question)
    return QueryResponse(**res)

@router.post("/sql-only", response_model=SQLOnlyResponse)
def get_sql_only(request: QueryRequest, controller: ChatController = Depends(get_chat_controller)):
    """Generate SQL statement only without database execution."""
    res = controller.generate_sql_only(request.question)
    return SQLOnlyResponse(**res)
