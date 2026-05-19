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

from .routes import chat_router, health_router, ingest_router
from .routes.chat_routes import get_rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: initialize service connections. Shut-down: close them."""
    print("[startup] Initializing SQL RAG Pipeline Service Layer...")
    # Trigger lazy instantiation
    service = get_rag_service()
    print("[startup] Services ready [OK]")
    yield
    print("[shutdown] Closing service connections...")
    service.close()
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
_FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
_CORS_ORIGINS = [_FRONTEND_URL] if _FRONTEND_URL != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind Routers
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(ingest_router)
