from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from ..config.config import MONGODB_URI, MONGODB_DB_NAME, HISTORY_LIMIT
from ..models.conversation_model import ChatMessage


class MongoRepository:
    """Handles all MongoDB operations for conversation memory."""

    def __init__(self):
        self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Ping to validate connection on startup without crashing if it fails
        try:
            self.client.admin.command("ping")
            print("[MongoRepository] Connected to MongoDB Atlas ✅")
        except Exception as e:
            print(f"[MongoRepository] ⚠️ Could not connect to MongoDB on startup: {e}")
            
        db = self.client[MONGODB_DB_NAME]
        self.collection = db["conversations"]
        # Index for fast session lookups
        self.collection.create_index("session_id")

    def save_message(self, user_id: str, session_id: str, question: str, sql: str, rows: list[dict]) -> None:
        """Append a Q&A turn to the conversation document."""
        # Validate data strictly before it goes into NoSQL
        msg_model = ChatMessage(
            question=question,
            sql=sql,
            row_preview=rows[:3],
            timestamp=datetime.utcnow()
        )
        
        self.collection.update_one(
            {"session_id": session_id, "user_id": user_id},
            {
                "$push": {"messages": msg_model.model_dump()},
                "$setOnInsert": {
                    "session_id": session_id, 
                    "user_id": user_id, 
                    "created_at": datetime.utcnow(),
                    "summary": question[:40] + ("..." if len(question) > 40 else "")
                }
            },
            upsert=True
        )

    def get_history(self, user_id: str, session_id: str) -> dict:
        """Retrieve the last N messages + rolling summary for a session."""
        doc = self.collection.find_one({"session_id": session_id, "user_id": user_id})
        if not doc:
            return {"summary": "", "messages": [], "total_messages": 0}

        # Return only the N most recent messages to prevent context explosion
        messages = doc.get("messages", [])
        recent_messages = messages[-HISTORY_LIMIT:] if HISTORY_LIMIT > 0 else messages

        return {
            "summary": doc.get("summary", ""),
            "messages": recent_messages,
            "total_messages": len(messages)
        }
        
    def get_all_messages(self, user_id: str, session_id: str) -> list[dict]:
        """Retrieve all messages for a session to generate a new summary."""
        doc = self.collection.find_one({"session_id": session_id, "user_id": user_id})
        if not doc:
            return []
        return doc.get("messages", [])

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """Fetch all sessions for a user, sorted by newest first, returning only metadata."""
        cursor = self.collection.find(
            {"user_id": user_id},
            {"session_id": 1, "summary": 1, "created_at": 1, "_id": 0}
        ).sort("created_at", -1)
        return list(cursor)
        
    def save_summary(self, user_id: str, session_id: str, summary: str) -> None:
        """Update the conversation summary for a session."""
        self.collection.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$set": {"summary": summary}},
            upsert=True
        )

    def check_connection(self) -> bool:
        """Ping MongoDB to verify connection is active."""
        self.client.admin.command("ping")
        return True

    def close(self) -> None:
        """Close the MongoDB connection cleanly."""
        self.client.close()
        print("[MongoRepository] MongoDB connection closed.")
