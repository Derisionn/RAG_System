import asyncpg
from ..config.config import AUTH_DB_URL
from ..models.user_model import UserInDB, UserSession
import uuid
from datetime import datetime

class AuthRepository:
    """Handles all PostgreSQL database operations for Authentication (users)."""

    def __init__(self):
        self.pool = None

    async def _get_pool(self):
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(AUTH_DB_URL)
                print("[AuthRepository] Connected to Supabase Auth DB ✅")
                await self._create_tables()
            except Exception as e:
                print(f"[AuthRepository] ⚠️ Could not connect to Supabase Auth DB: {e}")
        return self.pool

    async def _create_tables(self):
        pool = self.pool
        if not pool:
            return
        query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS user_sessions (
            id VARCHAR(36) PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            refresh_token_hash TEXT NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP WITH TIME ZONE,
            revoked_at TIMESTAMP WITH TIME ZONE,
            device_name VARCHAR(255),
            ip_address VARCHAR(45),
            user_agent TEXT
        );
        """
        async with pool.acquire() as conn:
            await conn.execute(query)

    async def create_user(self, email: str, hashed_password: str) -> UserInDB | None:
        """Create a new user. Returns the UserInDB model if successful, None if email exists."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        
        query = """
        INSERT INTO users (email, hashed_password)
        VALUES ($1, $2)
        RETURNING id, email, hashed_password, created_at;
        """
        try:
            async with pool.acquire() as conn:
                user = await conn.fetchrow(query, email, hashed_password)
                return UserInDB(**dict(user)) if user else None
        except asyncpg.exceptions.UniqueViolationError:
            # Email already exists
            return None

    async def get_user_by_email(self, email: str) -> UserInDB | None:
        """Fetch a user by their email address and return as a strict UserInDB model."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
            
        query = "SELECT id, email, hashed_password, created_at FROM users WHERE email = $1;"
        async with pool.acquire() as conn:
            user = await conn.fetchrow(query, email)
            return UserInDB(**dict(user)) if user else None

    async def get_user_by_id(self, user_id: int) -> UserInDB | None:
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
            
        query = "SELECT id, email, hashed_password, created_at FROM users WHERE id = $1;"
        async with pool.acquire() as conn:
            user = await conn.fetchrow(query, user_id)
            return UserInDB(**dict(user)) if user else None

    async def create_session(self, user_id: int, refresh_token_hash: str, expires_at: datetime, ip_address: str = None, user_agent: str = None) -> UserSession:
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
            
        session_id = str(uuid.uuid4())
        query = """
        INSERT INTO user_sessions (id, user_id, refresh_token_hash, expires_at, ip_address, user_agent)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *;
        """
        async with pool.acquire() as conn:
            session = await conn.fetchrow(query, session_id, user_id, refresh_token_hash, expires_at, ip_address, user_agent)
            return UserSession(**dict(session)) if session else None

    async def get_session_by_hash(self, refresh_token_hash: str) -> UserSession | None:
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
            
        query = "SELECT * FROM user_sessions WHERE refresh_token_hash = $1;"
        async with pool.acquire() as conn:
            session = await conn.fetchrow(query, refresh_token_hash)
            return UserSession(**dict(session)) if session else None

    async def update_session_last_used(self, session_id: str):
        pool = await self._get_pool()
        if not pool:
            return
        query = "UPDATE user_sessions SET last_used_at = CURRENT_TIMESTAMP WHERE id = $1;"
        async with pool.acquire() as conn:
            await conn.execute(query, session_id)

    async def revoke_session(self, refresh_token_hash: str):
        pool = await self._get_pool()
        if not pool:
            return
        query = "UPDATE user_sessions SET revoked_at = CURRENT_TIMESTAMP WHERE refresh_token_hash = $1;"
        async with pool.acquire() as conn:
            await conn.execute(query, refresh_token_hash)

    async def close(self):
        if self.pool:
            await self.pool.close()
            print("[AuthRepository] Supabase Auth DB pool closed. ✅")
