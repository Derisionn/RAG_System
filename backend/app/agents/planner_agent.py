from ..config.config import TOP_K_TABLES, TOP_K_COLUMNS

class PlannerAgent:
    def plan_schema(self, matches: list[dict]) -> tuple[list[str], list[dict]]:
        """
        Organize and filter Pinecone matches into a set of planned tables and columns
        for the context window of our reasoning agent.
        """
        import re
        
        tables = set()
        columns = []

        for match in matches:
            meta = match["metadata"]
            score = match["score"]
            
            raw_table_name = meta.get("table", "")
            # Map partition tables (e.g. public.payment_p2022_01) to parent (public.payment)
            table_name = re.sub(r'_p20\d{2}_\d{2}$', '', raw_table_name)
            
            # Map Pinecone schema fields back to standard keys
            mapped_meta = {
                "table_name": table_name,
                "column_name": meta.get("column"),
                "data_type": meta.get("data_type", "text"),
                "description": meta.get("description", "")
            }

            if meta.get("type") == "table":
                tables.add(table_name)
                print(f"  -> Planned Table: {table_name} (from {raw_table_name}, score={score:.4f})")
            else:
                columns.append(mapped_meta)
                tables.add(table_name)
                print(f"  -> Planned Column: {table_name}.{meta['column']} (from {raw_table_name}, score={score:.4f})")

            if len(tables) >= TOP_K_TABLES and len(columns) >= TOP_K_COLUMNS:
                break

        return list(tables), columns
