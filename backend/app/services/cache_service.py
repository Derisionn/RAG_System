import json
import hashlib
import redis
from typing import Optional, Any
from ..config.config import REDIS_URL

class CacheService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheService, cls).__new__(cls)
            cls._instance._init_redis()
        return cls._instance

    def _init_redis(self):
        self.enabled = False
        self.client = None
        if REDIS_URL and REDIS_URL.strip():
            try:
                # Use ssl_cert_reqs=None if needed for some free providers
                self.client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
                # Ping to check connection
                self.client.ping()
                self.enabled = True
                print("[CacheService] Redis connected successfully.")
            except Exception as e:
                print(f"[CacheService] Warning: Could not connect to Redis: {e}")
                self.enabled = False

    def generate_key(self, prefix: str, data: str) -> str:
        """Generate a SHA256 hash key for caching."""
        hashed = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return f"{prefix}:{hashed}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self.client:
            return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            print(f"[CacheService] Error getting key {key}: {e}")
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            val_str = json.dumps(value)
            self.client.setex(key, ttl_seconds, val_str)
            return True
        except Exception as e:
            print(f"[CacheService] Error setting key {key}: {e}")
            return False
