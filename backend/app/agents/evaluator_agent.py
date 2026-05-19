import sqlparse

class EvaluatorAgent:
    def validate_sql(self, sql: str) -> str | None:
        """
        Validate safety and basic syntax of a SQL query.
        Returns None if valid, or an error message string if invalid.
        """
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return "Query is empty or could not be parsed."

            # Ensure it is a read-only query (starts with SELECT or WITH)
            # Remove comments and whitespace first
            clean_sql = sqlparse.format(sql, strip_comments=True).strip().upper()
            
            # Simple keyword checks to prevent destructive queries
            allowed_prefixes = ("SELECT", "WITH")
            if not clean_sql.startswith(allowed_prefixes):
                return "Only read-only SELECT and WITH CTE queries are allowed."

            # Destructive keyword checks
            blacklisted_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
            for kw in blacklisted_keywords:
                if f" {kw} " in f" {clean_sql} ":
                    return f"Destructive keyword '{kw}' detected! Only SELECT queries are permitted."

            return None
        except Exception as e:
            return f"SQL syntax validation error: {str(e)}"
