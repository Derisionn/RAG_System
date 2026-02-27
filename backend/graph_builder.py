"""
graph_builder.py
────────────────
Loads schema.json and pushes table nodes and foreign key relationships into Neo4j.
This allows the RAG system to find join paths using Cypher queries.

Run:
    python graph_builder.py
"""

import json
import os
from neo4j import GraphDatabase
from tqdm import tqdm

from .config import (
    SCHEMA_JSON,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PWD
)

class Neo4jIngestor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_db(self):
        """Wipe the database before ingestion to avoid duplicates."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_table_nodes(self, tables):
        """Create a node for each table in the schema."""
        query = """
        UNWIND $tables AS table
        MERGE (t:Table {name: table.name})
        SET t.schema = table.schema
        """
        table_data = []
        for full_name in tables:
            schema_name, table_name = full_name.split('.')
            table_data.append({"name": full_name, "schema": schema_name})

        with self.driver.session() as session:
            session.run(query, tables=table_data)

    def create_relationships(self, fks):
        """Create :REFERENCES relationships between tables."""
        query = """
        MATCH (source:Table {name: $source_table})
        MATCH (target:Table {name: $target_table})
        MERGE (source)-[r:REFERENCES {
            source_col: $source_col, 
            target_col: $target_col,
            constraint_name: $constraint_name
        }]->(target)
        """
        with self.driver.session() as session:
            for fk in fks:
                session.run(query, 
                    source_table=fk["child_table"],
                    target_table=fk["parent_table"],
                    source_col=fk["child_column"],
                    target_col=fk["parent_column"],
                    constraint_name=fk["constraint_name"]
                )

def load_schema(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"schema.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("=" * 60)
    print(" Agentic Hybrid RAG - Neo4j Ingestion")
    print("=" * 60)

    try:
        schema = load_schema(SCHEMA_JSON)
        tables = list(schema["tables"].keys())
        fks = schema["foreign_key_relationships"]

        print(f"Connecting to Neo4j at {NEO4J_URI}...")
        print(f"Using Username: {NEO4J_USER}")
        print(f"Using Password: {NEO4J_PWD[:3]}...{NEO4J_PWD[-3:]}" if NEO4J_PWD else "Password: None")

        if not NEO4J_PWD or NEO4J_PWD == "your_password":
            print("[ERROR] Password is not set correctly in .env. Current value matches placeholder.")
            return

        ingestor = Neo4jIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PWD)

        print("Clearing existing data...")
        ingestor.clear_db()

        print(f"Creating {len(tables)} table nodes...")
        ingestor.create_table_nodes(tables)

        print(f"Creating {len(fks)} relationships...")
        ingestor.create_relationships(fks)

        ingestor.close()
        print("\n[SUCCESS] Neo4j ingestion complete.")

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")

if __name__ == "__main__":
    main()
