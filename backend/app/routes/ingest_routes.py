from fastapi import APIRouter, Depends
from ..controllers.ingest_controller import IngestController

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

def get_ingest_controller() -> IngestController:
    return IngestController()

@router.post("", status_code=202)
def trigger_ingestion(controller: IngestController = Depends(get_ingest_controller)):
    """Trigger schema ingestion pipeline."""
    return controller.trigger_ingest()
