"""
rag_pipeline.py
───────────────
The core retrieval segment of the Hybrid RAG system.
Focus: Step 1 - Semantic Retrieval (Pinecone)
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import google.generativeai as genai

from .config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PWD,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    CONNECTION_STRING,
    TOP_K_TABLES,
    TOP_K_COLUMNS,
    MAX_RETRIES,
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class SQLRAGPipeline:
    def __init__(self):
        print("Initializing Retrieval Engine (Pinecone + Neo4j)...")
        # 1. Load Embedding Model
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        # 2. Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vector_index = self.pc.Index(PINECONE_INDEX)

        # 3. Initialize Neo4j
        self.graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

        # 4. Initialize Gemini Model
        self.llm = genai.GenerativeModel(GEMINI_MODEL)

        # 5. Initialize SQL Engine
        self.sql_engine = create_engine(CONNECTION_STRING)

    def close(self):
        self.graph_driver.close()

    def retrieve_schema_elements(self, query: str):
        """
        STEP 1: Semantic Retrieval
        Search Pinecone for the most relevant tables and columns based on the query.
        """
        print(f"\n[Step 1] Embedding query: '{query}'")
        query_vec = self.model.encode([query]).tolist()
        
        print(f"[Step 1] Searching Pinecone index '{PINECONE_INDEX}'...")
        results = self.vector_index.query(
            vector=query_vec[0],
            top_k=20, # Fetch a reasonable chunk to filter
            include_metadata=True
        )
        
        tables = set()
        columns = []
        
        print(f"[Step 1] Found {len(results['matches'])} matches. Extracting relevant metadata...")
        for res in results['matches']:
            meta = res['metadata']
            score = res['score']
            
            if meta["type"] == "table":
                tables.add(meta["table"])
                print(f"  ▸ Table Match: {meta['table']} (Score: {score:.4f})")
            else:
                columns.append(meta)
                tables.add(meta["table"])
                print(f"  ▸ Column Match: {meta['table']}.{meta['column']} (Score: {score:.4f})")
                
            # Stop if we have enough context
            if len(tables) >= TOP_K_TABLES and len(columns) >= TOP_K_COLUMNS:
                break
                
        return list(tables), columns

    def get_join_paths(self, tables: list[str]):
        """
        STEP 2: Relational Retrieval (Graph)
        Query Neo4j for the shortest paths between all pairs of retrieved tables.
        """
        if len(tables) < 2:
            return []

        print(f"\n[Step 2] Finding join paths between {len(tables)} tables in Neo4j...")
        paths = []
        
        # We look for paths between all unique pairs of tables
        cypher = """
        MATCH (a:Table {name: $start_node}), (b:Table {name: $end_node})
        MATCH p = shortestPath((a)-[:REFERENCES*..3]-(b))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """

        with self.graph_driver.session() as session:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    result = session.run(cypher, start_node=tables[i], end_node=tables[j])
                    for record in result:
                        path = record["path_nodes"]
                        if path and len(path) > 1:
                            paths.append(path)
                            print(f"  ▸ Found Path: {' -> '.join(path)}")
        
        return paths

    def build_prompt(self, query: str, tables: list[str], columns: list[dict], paths: list[list[str]]):
        """
        STEP 3: Context Assembly
        Construct a prompt that gives the LLM all necessary schema and join context.
        """
        print("\n[Step 3] Assembling context into prompt...")
        
        # 1. Format Schema context
        context_schema = "RELEVANT SCHEMA ELEMENTS:\n"
        for tbl in tables:
            context_schema += f"- Table: {tbl}\n"
        
        context_schema += "\nRELEVANT COLUMNS:\n"
        # Only show a subset of columns to stay within token limits
        for col in columns[:TOP_K_COLUMNS]:
            context_schema += f"- {col['table']}.{col['column']} ({col['data_type']})\n"
            
        # 2. Format Join Paths
        context_joins = "\nSUGGESTED JOIN PATHS (Foreign Keys):\n"
        if paths:
            unique_paths = list(set([" -> ".join(p) for p in paths]))
            for p in unique_paths:
                context_joins += f"- {p}\n"
        else:
            context_joins += "- No direct relationship found in graph. Use common sense or partial joins if needed.\n"

        prompt = f"""
You are an expert SQL Generator for a Microsoft SQL Server database (AdventureWorks2019).
Given the user's natural language question, the relevant schema elements, and suggested join paths, generate a valid T-SQL query.

{context_schema}
{context_joins}

USER QUESTION: "{query}"

CONSTRAINTS:
1. Return ONLY the raw SQL code. No markdown formatting (no ```sql), no explanations.
2. Use fully qualified table names (Schema.Table).
3. Use JOINs based on the suggested paths where possible.
4. If a join path is A -> B -> C, use: FROM A JOIN B ON ... JOIN C ON ...
5. Use TOP for limiting results if appropriate.

SQL:"""
        return prompt

    def build_correction_prompt(self, original_prompt: str, failed_sql: str, error: str):
        """
        STEP 4b: Correction Prompt
        Builds a new prompt asking Gemini to fix the broken SQL using the error message.
        """
        return f"""
{original_prompt}

The SQL query you previously generated failed with the following error:

FAILED SQL:
{failed_sql}

ERROR MESSAGE:
{error}

Please analyse the error carefully and generate a corrected T-SQL query.
Return ONLY the raw SQL. No markdown, no explanation.

CORRECTED SQL:"""

    def generate_sql(self, query: str):
        """
        STEP 4: SQL Generation + Self-Correction Loop
        Full pipeline: Retrieve -> Path -> Prompt -> Gemini -> Execute -> Retry on failure.
        """
        # 1. Semantic Retrieval
        tables, columns = self.retrieve_schema_elements(query)

        # 2. Relational Retrieval
        paths = self.get_join_paths(tables)

        # 3. Context Assembly
        prompt = self.build_prompt(query, tables, columns, paths)

        # 4. First LLM Call
        print("\n[Step 4] Calling Gemini to generate SQL (Attempt 1)...")
        response = self.llm.generate_content(prompt)
        sql = response.text.strip().replace("```sql", "").replace("```", "").strip()

        # 5. Execute + Self-Correction Loop
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n[Step 5] Executing SQL (Attempt {attempt})...")
            df, error = self.execute_sql(sql)

            if error is None:
                # ✅ Success
                print(f"  ✅ Query succeeded on attempt {attempt}.")
                return sql, df, None

            # ❌ Failed — attempt correction if retries remain
            if attempt < MAX_RETRIES:
                print(f"  ⚠️  Attempt {attempt} failed. Asking Gemini to self-correct...")
                correction_prompt = self.build_correction_prompt(prompt, sql, error)
                response = self.llm.generate_content(correction_prompt)
                sql = response.text.strip().replace("```sql", "").replace("```", "").strip()
                print(f"\n[Step 4] Gemini correction (Attempt {attempt + 1}):")
                print(sql)
            else:
                print(f"  ❌ All {MAX_RETRIES} attempts exhausted.")

        return sql, None, error

    def execute_sql(self, sql: str):
        """
        STEP 5: SQL Execution
        Run the generated SQL on MSSQL and return a Pandas DataFrame.
        """
        try:
            df = pd.read_sql(sql, self.sql_engine)
            return df, None
        except Exception as e:
            print(f"  ❌ Execution Error: {str(e)[:300]}")
            return None, str(e)

def main():
    # Test the FULL RAG Pipeline with Self-Correction
    pipeline = SQLRAGPipeline()
    try:
        test_queries = [
            "Who are the top 5 customers by sales?",
            "How many distinct products are in the inventory?"
        ]

        for q in test_queries:
            print("\n" + "="*80)
            print(f"USER QUERY: {q}")
            print("="*80)

            # Full pipeline: Retrieve -> Generate -> Execute (with self-correction)
            sql, results, error = pipeline.generate_sql(q)

            print("\n--- FINAL SQL ---")
            print(sql)

            if error:
                print(f"\n❌ Failed after {MAX_RETRIES} attempts.")
                print(f"Last error: {error[:300]}")
            else:
                print("\n--- QUERY RESULTS ---")
                print(results.head())

    finally:
        pipeline.close()

if __name__ == "__main__":
    main()
