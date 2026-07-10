from fastapi import APIRouter, Depends, BackgroundTasks, Request
from ..schemas.request_models import QueryRequest
from ..schemas.response_models import QueryResponse, SQLOnlyResponse
from ..controllers.chat_controller import ChatController
from ..services.rag_service import RAGService
from ..repositories.mongodb_repository import MongoRepository
from ..auth.dependencies import get_current_user

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


from fastapi.responses import StreamingResponse

@router.post("", response_model=QueryResponse)
def execute_rag_query(req: Request, request: QueryRequest, background_tasks: BackgroundTasks, controller: ChatController = Depends(get_chat_controller), current_user: dict = Depends(get_current_user)):
    """Full RAG query execution against Supabase PostgreSQL database."""
    res = controller.execute_query(req, request.question, request.session_id, background_tasks, current_user["id"])
    return QueryResponse(**res)

@router.post("/stream")
def stream_rag_query(req: Request, request: QueryRequest, background_tasks: BackgroundTasks, controller: ChatController = Depends(get_chat_controller), current_user: dict = Depends(get_current_user)):
    """Stream execution progress for better UX using Server-Sent Events."""
    generator = controller.execute_query_stream(req, request.question, request.session_id, background_tasks, current_user["id"])
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/sql-only", response_model=SQLOnlyResponse)
def get_sql_only(request: QueryRequest, controller: ChatController = Depends(get_chat_controller), current_user: dict = Depends(get_current_user)):
    """Generate SQL statement only without database execution."""
    res = controller.generate_sql_only(request.question)
    return SQLOnlyResponse(**res)

@router.get("/sessions")
def get_user_sessions(mongo_repo: MongoRepository = Depends(get_mongo_repo), current_user: dict = Depends(get_current_user)):
    """Retrieve all chat sessions for the current user."""
    sessions = mongo_repo.get_user_sessions(str(current_user["id"]))
    # Convert datetime objects to ISO strings for JSON serialization
    for s in sessions:
        if "created_at" in s and s["created_at"]:
            s["created_at"] = s["created_at"].isoformat()
    return {"sessions": sessions}

@router.get("/sessions/{session_id}")
def get_session_history(session_id: str, mongo_repo: MongoRepository = Depends(get_mongo_repo), current_user: dict = Depends(get_current_user)):
    """Retrieve all historical messages for a specific session."""
    messages = mongo_repo.get_all_messages(str(current_user["id"]), session_id)
    # Convert datetime objects to ISO strings
    for m in messages:
        if "timestamp" in m and m["timestamp"]:
            m["timestamp"] = m["timestamp"].isoformat()
    return {"messages": messages}
