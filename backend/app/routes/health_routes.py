from fastapi import APIRouter, Depends
from ..schemas.response_models import HealthResponse
from ..controllers.health_controller import HealthController
from .chat_routes import get_rag_service

router = APIRouter(prefix="/health", tags=["System"])

def get_health_controller(service = Depends(get_rag_service)) -> HealthController:
    return HealthController(service)

@router.get("", response_model=HealthResponse)
def get_health(controller: HealthController = Depends(get_health_controller)):
    """Fetch status reports for all downstream services."""
    res = controller.get_health_status()
    return HealthResponse(**res)
