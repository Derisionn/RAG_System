from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from ..config.config import MONGODB_URI, MONGODB_DB_NAME, HISTORY_LIMIT


class MongoRepository:
    """Handles all MongoDB operations for conversation memory."""

    def __init__(self):
        self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Ping to validate connection on startup
        self.client.admin.command("ping")
        db = self.client[MONGODB_DB_NAME]
        self.collection = db["conversations"]
        # Index for fast session lookups
        self.collection.create_index("session_id")
        print("[MongoRepository] Connected to MongoDB Atlas ✅")

    def save_message(self, session_id: str, question: str, sql: str, rows: list[dict]) -> None:
        """Append a Q&A turn to the conversation document."""
        message = {
            "question": question,
            "sql": sql,
            "row_preview": rows[:3],   # store first 3 rows as context preview
            "timestamp": datetime.now(timezone.utc),
        }
        self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message},
                "$setOnInsert": {"session_id": session_id, "created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

    def get_history(self, session_id: str) -> list[dict]:
        """Retrieve the last HISTORY_LIMIT messages for a session."""
        doc = self.collection.find_one({"session_id": session_id})
        if not doc:
            return []
        messages = doc.get("messages", [])
        # Return only the last N messages
        return messages[-HISTORY_LIMIT:]

    def close(self) -> None:
        """Close the MongoDB connection cleanly."""
        self.client.close()
        print("[MongoRepository] MongoDB connection closed.")
