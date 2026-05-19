"""
config.py — Central configuration for the Agentic Hybrid RAG System.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

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

# Schemas to index (set to ["public"] for default Supabase, or keep AdventureWorks schemas if migrated)
TARGET_SCHEMAS = os.getenv("TARGET_SCHEMAS", "public,Sales,Production,HumanResources,Purchasing,Person").split(",")

# Pinecone Vector DB
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_key")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "adventureworks-schema")

# Neo4j Graph DB
NEO4J_URI   = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD   = os.getenv("NEO4J_PASSWORD", "your_password")

# Schemas to index (set to None to index ALL schemas)
TARGET_SCHEMAS = ["Sales", "Production", "HumanResources", "Purchasing", "Person"]

# ── Paths ─────────────────────────────────────────────────────────────────────
# backend/ lives one level inside the project root — point BASE_DIR at the root
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_JSON     = os.path.join(BASE_DIR, "schema.json")
# FAISS paths are now deprecated in favor of Pinecone
# FAISS_DIR       = os.path.join(BASE_DIR, "faiss_index")
# FAISS_INDEX     = os.path.join(FAISS_DIR, "index.faiss")
# FAISS_META      = os.path.join(FAISS_DIR, "metadata.pkl")
# GRAPH_PKL is deprecated in favor of Neo4j, but we'll keep the constant for now
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

# ── LLM (Google Gemini) ───────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-flash-latest"
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")  # set in env

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3    # max SQL self-correction attempts
