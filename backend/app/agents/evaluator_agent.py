import sqlparse

class EvaluatorAgent:
    def validate_sql(self, sql: str) -> str | None:
        """
        Validate safety of a SQL query using deep AST parsing.
        Returns None if valid, or an error message string if invalid.
        """
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return "Query is empty or could not be parsed."

            # Ensure the statement starts with a read-only keyword
            clean_sql = sqlparse.format(sql, strip_comments=True).strip().upper()
            allowed_prefixes = ("SELECT", "WITH")
            if not clean_sql.startswith(allowed_prefixes):
                return "Only read-only SELECT and WITH CTE queries are allowed."

            # Deep token analysis
            destructive_keywords = {
                "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", 
                "TRUNCATE", "CREATE", "EXEC", "EXECUTE", "CALL", "DO",
                "GRANT", "REVOKE"
            }

            for statement in parsed:
                # get_type() usually returns the first keyword's category (e.g. 'SELECT')
                stmt_type = statement.get_type().upper()
                if stmt_type in destructive_keywords:
                    return f"Destructive statement type '{stmt_type}' detected."

                # Traverse all tokens in the AST to catch hidden destructive commands
                def check_tokens(tokens):
                    for token in tokens:
                        if token.is_group:
                            error = check_tokens(token.tokens)
                            if error: return error
                        elif token.ttype in sqlparse.tokens.Keyword or token.ttype in sqlparse.tokens.Keyword.DML or token.ttype in sqlparse.tokens.Keyword.DDL:
                            val = token.value.upper()
                            if val in destructive_keywords:
                                return f"Destructive keyword '{val}' detected in query structure."
                    return None
                
                err = check_tokens(statement.tokens)
                if err:
                    return err

            return None
        except Exception as e:
            return f"SQL syntax validation error: {str(e)}"
