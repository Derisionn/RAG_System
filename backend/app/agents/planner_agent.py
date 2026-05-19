from ..config.config import TOP_K_TABLES, TOP_K_COLUMNS

class PlannerAgent:
    def plan_schema(self, matches: list[dict]) -> tuple[list[str], list[dict]]:
        """
        Organize and filter Pinecone matches into a set of planned tables and columns
        for the context window of our reasoning agent.
        """
        tables = set()
        columns = []

        for match in matches:
            meta = match["metadata"]
            score = match["score"]
            
            # Map Pinecone schema fields back to standard keys
            mapped_meta = {
                "table_name": meta.get("table"),
                "column_name": meta.get("column"),
                "data_type": meta.get("type", "text"),
                "description": meta.get("description", "")
            }

            if meta.get("type") == "table":
                tables.add(meta["table"])
                print(f"  -> Planned Table: {meta['table']} (score={score:.4f})")
            else:
                columns.append(mapped_meta)
                tables.add(meta["table"])
                print(f"  -> Planned Column: {meta['table']}.{meta['column']} (score={score:.4f})")

            if len(tables) >= TOP_K_TABLES and len(columns) >= TOP_K_COLUMNS:
                break

        return list(tables), columns
