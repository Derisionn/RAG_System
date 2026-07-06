import os
import sys
import io
from contextlib import asynccontextmanager

# Force UTF-8 stdout/stderr on Windows to prevent charmap encoding errors
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat_router, health_router, auth_router
from .routes.chat_routes import get_rag_service, get_mongo_repo

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: initialize service connections. Shut-down: close them."""
    print("[startup] Initializing SQL RAG Pipeline Service Layer...")
    # Lazy instantiation deferred to the first request to prevent Render Port Scan Timeout
    # service = get_rag_service()
    # mongo = get_mongo_repo()
    print("[startup] Services will load lazily on first request [OK]")
    yield
    print("[shutdown] Closing service connections...")
    get_rag_service().close()
    get_mongo_repo().close()
    print("[shutdown] Service cleanup done.")


app = FastAPI(
    title="SQL RAG System Clean API",
    description=(
        "Layered Controller-Service-Repository implementation of our SQL RAG System "
        "generating compatible PostgreSQL queries on Supabase with self-correction capabilities."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware setup
_FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
_CORS_ORIGINS = [_FRONTEND_URL] if _FRONTEND_URL != "*" else ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(health_router)
