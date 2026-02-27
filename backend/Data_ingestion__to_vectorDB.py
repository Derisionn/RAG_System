"""
Data_ingestion__to_vectorDB.py
──────────────────────────────
Loads schema.json produced by schema_extractor.py, converts every table and
column into a rich textual description, and upserts them to Pinecone using
LangChain's PineconeVectorStore + HuggingFaceEmbeddings.

Run:
    python -m backend.Data_ingestion__to_vectorDB
  or from project root:
    python backend/Data_ingestion__to_vectorDB.py
"""

from __future__ import annotations

import json
import os
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from .config import (
    SCHEMA_JSON,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    PINECONE_API_KEY,
    PINECONE_INDEX,
)


# ── 1. Load schema ────────────────────────────────────────────────────────────

def load_schema(path: str) -> dict:
    """Load the JSON schema produced by schema_extractor.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"schema.json not found at {path}.\n"
            "Run schema_extractor.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 2. Build LangChain Documents ──────────────────────────────────────────────

def build_documents(schema: dict) -> list[Document]:
    """
    Convert each table and column into a LangChain Document.
    page_content  → the text that gets embedded
    metadata      → stored alongside the vector in Pinecone
    """
    docs: list[Document] = []

    for full_table, meta in schema["tables"].items():
        schema_name, table_name = full_table.split(".")
        columns     = meta.get("columns", [])
        pk_cols     = meta.get("primary_keys", [])
        col_summary = ", ".join(
            f"{c['name']} ({c['data_type']})" for c in columns
        )

        # ── Table-level document ──────────────────────────────────────────────
        table_text = (
            f"Table: {full_table}. "
            f"Schema: {schema_name}. "
            f"Columns: {col_summary}. "
            f"Primary keys: {', '.join(pk_cols) or 'none'}."
        )
        docs.append(Document(
            page_content=table_text,
            metadata={
                "type":      "table",
                "schema":    schema_name,
                "table":     full_table,
                "column":    "",
                "data_type": "",
                "text":      table_text,
                "doc_id":    f"table_{full_table}",
            }
        ))

        # ── Column-level documents ────────────────────────────────────────────
        for col in columns:
            is_pk    = col["name"] in pk_cols
            col_text = (
                f"Column: {col['name']} in table {full_table}. "
                f"Data type: {col['data_type']}. "
                f"Nullable: {col.get('nullable', True)}. "
                + ("This is a primary key column. " if is_pk else "")
                + f"Table schema: {schema_name}."
            )
            docs.append(Document(
                page_content=col_text,
                metadata={
                    "type":      "column",
                    "schema":    schema_name,
                    "table":     full_table,
                    "column":    col["name"],
                    "data_type": col["data_type"],
                    "text":      col_text,
                    "doc_id":    f"col_{full_table}_{col['name']}",
                }
            ))

    return docs


# ── 3. Ensure Pinecone index exists ───────────────────────────────────────────

def ensure_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        print(f"  Index '{PINECONE_INDEX}' already exists — reusing.")
        return

    print(f"  Creating index '{PINECONE_INDEX}'...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(PINECONE_INDEX).status["ready"]:
        time.sleep(1)
    print(f"  Index '{PINECONE_INDEX}' ready.")


# ── 4. Upsert via LangChain ───────────────────────────────────────────────────

def upsert_documents(docs: list[Document]) -> None:
    """
    Embed all documents with HuggingFaceEmbeddings and upsert to Pinecone
    using LangChain's PineconeVectorStore — no manual batching required.
    """
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Initialising Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc)

    print(f"Upserting {len(docs)} documents to '{PINECONE_INDEX}'...")

    # LangChain handles batching + embedding + upsert automatically
    batch_size = 50
    for i in tqdm(range(0, len(docs), batch_size), desc="Upserting batches"):
        batch = docs[i : i + batch_size]
        ids   = [doc.metadata["doc_id"] for doc in batch]
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name=PINECONE_INDEX,
            ids=ids,
        )

    print(f"\n✅  Successfully upserted {len(docs)} documents to Pinecone.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(" Agentic Hybrid RAG — LangChain Pinecone Ingestion")
    print("=" * 60)

    schema = load_schema(SCHEMA_JSON)
    docs   = build_documents(schema)
    print(f"\nGenerated {len(docs)} documents from {len(schema['tables'])} tables.")

    upsert_documents(docs)
    print("\n✅  Ingestion complete. Pinecone is ready.")


if __name__ == "__main__":
    main()