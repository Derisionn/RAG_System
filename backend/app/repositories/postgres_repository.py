import pandas as pd
from sqlalchemy import create_engine, text
from ..config.config import CONNECTION_STRING

from ..services.cache_service import CacheService
import json

class PostgresRepository:
    def __init__(self):
        self.engine = create_engine(CONNECTION_STRING)
        self.cache = CacheService()

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a Pandas DataFrame."""
        cache_key = self.cache.generate_key("sql_exec", sql.strip())
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            print("  [PostgresRepository] Cache HIT for SQL execution.")
            return pd.DataFrame.from_records(cached_data)

        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
            
        try:
            # Serialize to JSON and cache for 5 minutes
            records = json.loads(df.to_json(orient="records", date_format="iso"))
            self.cache.set(cache_key, records, ttl_seconds=300)
        except Exception as e:
            print(f"  [PostgresRepository] Warning: failed to cache results: {e}")
            
        return df

    def check_connection(self) -> bool:
        """Check if Supabase connection is healthy."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def close(self):
        """Clean up engine resource pool."""
        self.engine.dispose()
