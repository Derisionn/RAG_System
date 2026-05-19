class MemoryService:
    """
    Session memory cache service.
    Can be used to store active chat/history contexts.
    """
    def __init__(self):
        self._sessions = {}

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
