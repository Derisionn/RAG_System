"""
config.py — Central configuration for the Agentic Hybrid RAG System.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# ── Database ──────────────────────────────────────────────────────────────────
# Prioritize Supabase / PostgreSQL connection strings.
# Supabase URI format: postgresql+psycopg2://postgres:<password>@db.<project>.supabase.co:5432/postgres

_FALLBACK_CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"
)

CONNECTION_STRING = (
    os.getenv("SUPABASE_CONNECTION_STRING")
    or os.getenv("DATABASE_URL")
    or os.getenv("MSSQL_CONNECTION_STRING")
    or _FALLBACK_CONNECTION_STRING
)

# Schemas to index (default: public, Sales, Production, etc.)
TARGET_SCHEMAS = os.getenv("TARGET_SCHEMAS", "public,Sales,Production,HumanResources,Purchasing,Person").split(",")

# Pinecone Vector DB
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_key")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "dvd-rental-schema")

# Neo4j Graph DB
NEO4J_URI   = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD   = os.getenv("NEO4J_PASSWORD", "your_password")

# ── Paths ─────────────────────────────────────────────────────────────────────
# app/config/ lives three levels inside the project root — point BASE_DIR at the root
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHEMA_JSON     = os.path.join(BASE_DIR, "schema.json")
GRAPH_PKL       = os.path.join(BASE_DIR, "db_graph.pkl")

# ── Embedding Model ───────────────────────────────────────────────────────────
# Using HuggingFace Inference API — same model as the Pinecone index, zero local torch.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HF model ID
EMBEDDING_DIM   = 384                                        # output dimension
HF_API_TOKEN    = os.getenv("HF_API_TOKEN", "")              # set in Render env vars
HF_API_URL      = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_TABLES    = 5    # number of tables returned by semantic search
TOP_K_COLUMNS   = 10   # number of columns returned by semantic search

# =============================================================================
# Hugging Face Configuration (replaces Gemini)
# =============================================================================
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")  # set in env

# ── Google OAuth 2.0 ──────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3    # max SQL self-correction attempts

# ── MongoDB & Memory System ───────────────────────────────────────────────────
MONGODB_URI      = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME  = os.getenv("MONGODB_DB_NAME", "rag_system")
HISTORY_LIMIT    = 10   # number of recent messages to inject (5 Q&A pairs)
SUMMARY_TRIGGER_THRESHOLD = 15 # total messages before generating a new summary
PINECONE_HISTORY_NAMESPACE      = "chat-history"       # SQL Q&A pairs
PINECONE_CHAT_HISTORY_NAMESPACE = "chat-conversations"  # Conversational Q&A pairs

# ── Authentication ──────────────────────────────────────────────────────────────
AUTH_DB_URL = os.getenv("AUTH_DB_URL", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080")) # 7 days

# ── Email Service ─────────────────────────────────────────────────────────────
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "noreply@example.com")
print(f"DEBUG CONFIG: SENDGRID_API_KEY={SENDGRID_API_KEY[:10] if SENDGRID_API_KEY else None}")

