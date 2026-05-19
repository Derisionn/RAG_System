import pandas as pd
from sqlalchemy import create_engine, text
from ..config.config import CONNECTION_STRING

class PostgresRepository:
    def __init__(self):
        self.engine = create_engine(CONNECTION_STRING)

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a Pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn)

    def check_connection(self) -> bool:
        """Check if Supabase connection is healthy."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def close(self):
        """Clean up engine resource pool."""
        self.engine.dispose()
