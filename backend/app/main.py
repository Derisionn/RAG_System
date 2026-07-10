import os
import sys
import io
import time
from contextlib import asynccontextmanager

# Force UTF-8 stdout/stderr on Windows to prevent charmap encoding errors
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
from fastapi import FastAPI, Request
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

import logging
# Disable Uvicorn's default access log so we don't get duplicate lines
logging.getLogger("uvicorn.access").disabled = True

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time_ms = (time.time() - start_time) * 1000
    
    # Get standard access log details
    client_host = request.client.host if request.client else "127.0.0.1"
    client_port = request.client.port if request.client else "0"
    http_version = request.scope.get("http_version", "1.1")
    
    import http
    try:
        status_phrase = http.HTTPStatus(response.status_code).phrase
    except Exception:
        status_phrase = ""
        
    route_str = f'{client_host}:{client_port} - "{request.method} {request.url.path} HTTP/{http_version}" {response.status_code} {status_phrase}'
    
    # Store these in request state so controllers can inject them into the final box
    request.state.route_str = route_str
    request.state.handshake_ms = process_time_ms
    
    # We remove the standalone logger.info here so we only print the consolidated Box at the end!
    
    response.headers["X-Process-Time"] = str(process_time_ms / 1000.0)
    return response

# Bind Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(health_router)
