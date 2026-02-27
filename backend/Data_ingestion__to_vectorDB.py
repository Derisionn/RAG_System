"""
Data_ingestion__to_vectorDB.py
──────────────────────────────
Loads schema.json produced by schema_extractor.py, converts every table and
column into a rich textual description, embeds them with sentence-transformers
(all-MiniLM-L6-v2), and upserts them to Pinecone.

Run:
    python Data_ingestion__to_vectorDB.py
"""

import json
import os
import time

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
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


# ── 2. Build text chunks ──────────────────────────────────────────────────────
def build_text_chunks(schema: dict) -> list[dict]:
    """
    Convert each table into a high-level description AND each column into its
    own chunk.
    """
    chunks: list[dict] = []

    for full_table, meta in schema["tables"].items():
        schema_name, table_name = full_table.split(".")
        columns     = meta.get("columns", [])
        pk_cols     = meta.get("primary_keys", [])
        col_summary = ", ".join(
            f"{c['name']} ({c['data_type']})"
            for c in columns
        )

        # ── Table-level chunk ──────────────────────────────────────────────
        table_text = (
            f"Table: {full_table}. "
            f"Schema: {schema_name}. "
            f"Columns: {col_summary}. "
            f"Primary keys: {', '.join(pk_cols) or 'none'}."
        )
        chunks.append({
            "id": f"table_{full_table}",
            "text": table_text,
            "metadata": {
                "type":      "table",
                "schema":    schema_name,
                "table":     full_table,
                "column":    "",
                "data_type": "",
                "text":      table_text, # Pinecone metadata allows querying text
            }
        })

        # ── Column-level chunks ────────────────────────────────────────────
        for col in columns:
            is_pk = col["name"] in pk_cols
            col_text = (
                f"Column: {col['name']} in table {full_table}. "
                f"Data type: {col['data_type']}. "
                f"Nullable: {col.get('nullable', True)}. "
                + ("This is a primary key column. " if is_pk else "")
                + f"Table schema: {schema_name}."
            )
            chunks.append({
                "id": f"col_{full_table}_{col['name']}",
                "text": col_text,
                "metadata": {
                    "type":      "column",
                    "schema":    schema_name,
                    "table":     full_table,
                    "column":    col["name"],
                    "data_type": col["data_type"],
                    "text":      col_text,
                }
            })

    return chunks


# ── 3. Upsert to Pinecone ─────────────────────────────────────────────────────
def upsert_to_pinecone(chunks: list[dict], model_name: str):
    """Embed chunks and upsert to Pinecone in batches."""
    print(f"\nInitializing Pinecone index '{PINECONE_INDEX}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Delete if exists to refresh, or just create
    if PINECONE_INDEX in [idx.name for idx in pc.list_indexes()]:
        print(f"Index '{PINECONE_INDEX}' already exists.")
    else:
        print(f"Creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX).status['ready']:
            time.sleep(1)

    index = pc.Index(PINECONE_INDEX)

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Embedding and upserting {len(chunks)} chunks in batches...")
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        ids = [chunk["id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        
        # Embed
        embeddings = model.encode(texts).tolist()
        
        # Prepare for upsert
        to_upsert = list(zip(ids, embeddings, metadatas))
        
        # Upsert
        index.upsert(vectors=to_upsert)

    print(f"\n✅ Successfully upserted {len(chunks)} vectors to Pinecone.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Agentic Hybrid RAG — Pinecone Ingestion")
    print("=" * 60)

    schema  = load_schema(SCHEMA_JSON)
    chunks  = build_text_chunks(schema)
    print(f"\nGenerated {len(chunks)} text chunks "
          f"from {len(schema['tables'])} tables.")

    upsert_to_pinecone(chunks, EMBEDDING_MODEL)

    print("\n✅ Ingestion complete. Pinecone is ready.")


if __name__ == "__main__":
    main()