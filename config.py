"""
config.py — Central configuration for the Agentic Hybrid RAG System.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_SERVER   = "laptop"                  # SQL Server instance name / hostname
DB_NAME     = "AdventureWorks2019"
DB_DRIVER   = "ODBC+Driver+17+for+SQL+Server"

# Windows Integrated Auth (no username/password required)
CONNECTION_STRING = (
    f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}"
    f"?driver={DB_DRIVER}"
    "&trusted_connection=yes"
)

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
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SCHEMA_JSON     = os.path.join(BASE_DIR, "schema.json")
# FAISS paths are now deprecated in favor of Pinecone
# FAISS_DIR       = os.path.join(BASE_DIR, "faiss_index")
# FAISS_INDEX     = os.path.join(FAISS_DIR, "index.faiss")
# FAISS_META      = os.path.join(FAISS_DIR, "metadata.pkl")
# GRAPH_PKL is deprecated in favor of Neo4j, but we'll keep the constant for now
GRAPH_PKL       = os.path.join(BASE_DIR, "db_graph.pkl")

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model
EMBEDDING_DIM   = 384                   # dimension for all-MiniLM-L6-v2

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_TABLES    = 5    # number of tables returned by semantic search
TOP_K_COLUMNS   = 10   # number of columns returned by semantic search

# ── LLM (Google Gemini) ───────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-flash-latest"
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")  # set in env

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3    # max SQL self-correction attempts
