from fastapi import APIRouter, Depends
from ..schemas.request_models import QueryRequest
from ..schemas.response_models import QueryResponse, SQLOnlyResponse
from ..controllers.chat_controller import ChatController
from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository

router = APIRouter(prefix="/query", tags=["RAG"])

# ── Shared singletons (created once, reused for all requests) ─────────────────
_rag_service: RAGService | None = None
_mongo_repo: MongoRepository | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_mongo_repo() -> MongoRepository:
    global _mongo_repo
    if _mongo_repo is None:
        _mongo_repo = MongoRepository()
    return _mongo_repo


def get_chat_controller(
    rag_service: RAGService = Depends(get_rag_service),
    mongo_repo: MongoRepository = Depends(get_mongo_repo),
) -> ChatController:
    return ChatController(rag_service, mongo_repo)


@router.post("", response_model=QueryResponse)
def execute_rag_query(request: QueryRequest, controller: ChatController = Depends(get_chat_controller)):
    """Full RAG query execution against Supabase PostgreSQL database."""
    res = controller.execute_query(request.question, request.session_id)
    return QueryResponse(**res)


@router.post("/sql-only", response_model=SQLOnlyResponse)
def get_sql_only(request: QueryRequest, controller: ChatController = Depends(get_chat_controller)):
    """Generate SQL statement only without database execution."""
    res = controller.generate_sql_only(request.question)
    return SQLOnlyResponse(**res)
