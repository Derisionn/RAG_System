# Render entrypoint — re-exports the FastAPI app so uvicorn can find it
# from the project root without needing backend.api:app module path.
from backend.api import app  # noqa: F401
