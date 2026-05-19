class RedisRepository:
    """
    A placeholder/stub Redis repository.
    Can be easily swapped with actual Redis client library in production.
    Currently acts as a local dict cache.
    """
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self._cache[key] = value

    def check_connection(self) -> str:
        """Check cache/stub status for health check."""
        return "ok — local cache active"
