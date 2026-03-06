"""
Data_ingestion__to_vectorDB.py
──────────────────────────────
Loads schema.json produced by schema_extractor.py, converts every table and
column into a rich textual description, and upserts them to Pinecone.

Embeddings are generated via the HuggingFace Inference API
(sentence-transformers/all-MiniLM-L6-v2, 384-dim) — no local torch/model needed.

Run:
    python -m backend.Data_ingestion__to_vectorDB
  or from project root:
    python backend/Data_ingestion__to_vectorDB.py
"""

from __future__ import annotations

import json
import os
import time

import requests
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from .config import (
    SCHEMA_JSON,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    HF_API_TOKEN,
    HF_API_URL,
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


# ── 2. Build documents ────────────────────────────────────────────────────────

def build_documents(schema: dict) -> list[dict]:
    """
    Convert each table and column into a dict with:
      text     → the string to embed
      metadata → stored alongside the vector in Pinecone
      doc_id   → unique vector ID
    """
    docs: list[dict] = []

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
        docs.append({
            "text":    table_text,
            "doc_id":  f"table_{full_table}",
            "metadata": {
                "type":      "table",
                "schema":    schema_name,
                "table":     full_table,
                "column":    "",
                "data_type": "",
                "text":      table_text,
            }
        })

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
            docs.append({
                "text":    col_text,
                "doc_id":  f"col_{full_table}_{col['name']}",
                "metadata": {
                    "type":      "column",
                    "schema":    schema_name,
                    "table":     full_table,
                    "column":    col["name"],
                    "data_type": col["data_type"],
                    "text":      col_text,
                }
            })

    return docs


# ── 3. HF Inference API embedding ─────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using HuggingFace Inference API.
    Returns list of 384-dim float vectors (same model as Pinecone index).
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    resp = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # HF feature-extraction for a list input returns list[list[float]]
    return data


# ── 4. Ensure Pinecone index exists ───────────────────────────────────────────

def ensure_index(pc: Pinecone) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        print(f"  Index '{PINECONE_INDEX}' already exists — reusing.")
        return

    print(f"  Creating index '{PINECONE_INDEX}' (dim={EMBEDDING_DIM}, metric=cosine)...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(PINECONE_INDEX).status["ready"]:
        time.sleep(1)
    print(f"  Index '{PINECONE_INDEX}' ready ✅")


# ── 5. Upsert ─────────────────────────────────────────────────────────────────

def upsert_documents(docs: list[dict]) -> None:
    """Embed all documents via HF API and upsert to Pinecone in batches."""
    print(f"\nEmbedding model : {EMBEDDING_MODEL} (via HuggingFace Inference API)")
    print(f"Vector dimension: {EMBEDDING_DIM}")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc)
    index = pc.Index(PINECONE_INDEX)

    print(f"Upserting {len(docs)} documents to '{PINECONE_INDEX}'...")

    batch_size = 32  # conservative for HF free-tier rate limits
    for i in tqdm(range(0, len(docs), batch_size), desc="Upserting batches"):
        batch  = docs[i : i + batch_size]
        texts  = [d["text"] for d in batch]
        vecs   = embed_batch(texts)

        vectors = [
            {"id": d["doc_id"], "values": v, "metadata": d["metadata"]}
            for d, v in zip(batch, vecs)
        ]
        index.upsert(vectors=vectors)

    print(f"\n✅  Successfully upserted {len(docs)} documents to Pinecone.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(" Agentic Hybrid RAG — HF Inference API Ingestion")
    print("=" * 60)

    schema = load_schema(SCHEMA_JSON)
    docs   = build_documents(schema)
    print(f"\nGenerated {len(docs)} documents from {len(schema['tables'])} tables.")

    upsert_documents(docs)
    print("\n✅  Ingestion complete. Pinecone is ready.")


if __name__ == "__main__":
    main()