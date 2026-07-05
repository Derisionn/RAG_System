import asyncio
import asyncpg
from ..config.config import AUTH_DB_URL
from ..models.user_model import UserInDB, UserSession
import uuid
from datetime import datetime

class AuthRepository:
    """Handles all PostgreSQL database operations for Authentication (users)."""

    def __init__(self):
        self.pool = None
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        # Lock prevents race condition where two concurrent first-requests both create a pool
        async with self._lock:
            if not self.pool:
                try:
                    self.pool = await asyncpg.create_pool(AUTH_DB_URL, statement_cache_size=0)
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
            hashed_password VARCHAR(255),
            display_name VARCHAR(100),
            google_id VARCHAR(255) UNIQUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Safe migrations for existing tables
        ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(100);
        ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id VARCHAR(255) UNIQUE;
        ALTER TABLE users ALTER COLUMN hashed_password DROP NOT NULL;

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
        CREATE TABLE IF NOT EXISTS email_otps (
            email VARCHAR(255) PRIMARY KEY,
            otp VARCHAR(10) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            is_verified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        async with pool.acquire() as conn:
            await conn.execute(query)

    # ── User CRUD ──────────────────────────────────────────────────────────────

    async def create_user(self, email: str, hashed_password: str, display_name: str = None) -> UserInDB | None:
        """Create a new email/password user."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = """
        INSERT INTO users (email, hashed_password, display_name)
        VALUES ($1, $2, $3)
        RETURNING id, email, hashed_password, display_name, google_id, created_at;
        """
        try:
            async with pool.acquire() as conn:
                user = await conn.fetchrow(query, email, hashed_password, display_name)
                return UserInDB(**dict(user)) if user else None
        except asyncpg.exceptions.UniqueViolationError:
            return None

    async def create_oauth_user(self, email: str, display_name: str, google_id: str) -> UserInDB | None:
        """Create a new user via Google OAuth (no password)."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = """
        INSERT INTO users (email, display_name, google_id)
        VALUES ($1, $2, $3)
        RETURNING id, email, hashed_password, display_name, google_id, created_at;
        """
        try:
            async with pool.acquire() as conn:
                user = await conn.fetchrow(query, email, display_name, google_id)
                return UserInDB(**dict(user)) if user else None
        except asyncpg.exceptions.UniqueViolationError:
            return None

    async def get_user_by_email(self, email: str) -> UserInDB | None:
        """Fetch a user by email."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "SELECT id, email, hashed_password, display_name, google_id, created_at FROM users WHERE email = $1;"
        async with pool.acquire() as conn:
            user = await conn.fetchrow(query, email)
            return UserInDB(**dict(user)) if user else None

    async def get_user_by_id(self, user_id: int) -> UserInDB | None:
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "SELECT id, email, hashed_password, display_name, google_id, created_at FROM users WHERE id = $1;"
        async with pool.acquire() as conn:
            user = await conn.fetchrow(query, user_id)
            return UserInDB(**dict(user)) if user else None

    async def get_user_by_google_id(self, google_id: str) -> UserInDB | None:
        """Fetch a user by their Google sub ID."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "SELECT id, email, hashed_password, display_name, google_id, created_at FROM users WHERE google_id = $1;"
        async with pool.acquire() as conn:
            user = await conn.fetchrow(query, google_id)
            return UserInDB(**dict(user)) if user else None

    async def link_google_id(self, user_id: int, google_id: str):
        """Attach a google_id to an existing email/password account."""
        pool = await self._get_pool()
        if not pool:
            return
        query = "UPDATE users SET google_id = $1 WHERE id = $2;"
        async with pool.acquire() as conn:
            await conn.execute(query, google_id, user_id)

    # ── Sessions ───────────────────────────────────────────────────────────────

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

    async def update_password(self, user_id: int, hashed_password: str):
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "UPDATE users SET hashed_password = $1 WHERE id = $2;"
        async with pool.acquire() as conn:
            await conn.execute(query, hashed_password, user_id)

    # ── OTPs ──────────────────────────────────────────────────────────────────

    async def create_or_update_otp(self, email: str, otp: str, expires_at: datetime):
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = """
        INSERT INTO email_otps (email, otp, expires_at, is_verified)
        VALUES ($1, $2, $3, FALSE)
        ON CONFLICT (email) DO UPDATE SET 
            otp = EXCLUDED.otp,
            expires_at = EXCLUDED.expires_at,
            is_verified = FALSE,
            created_at = CURRENT_TIMESTAMP;
        """
        async with pool.acquire() as conn:
            await conn.execute(query, email, otp, expires_at)

    async def get_otp(self, email: str):
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "SELECT otp, expires_at, is_verified FROM email_otps WHERE email = $1;"
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, email)

    async def mark_otp_verified(self, email: str):
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "UPDATE email_otps SET is_verified = TRUE WHERE email = $1;"
        async with pool.acquire() as conn:
            await conn.execute(query, email)

    async def delete_otp(self, email: str):
        pool = await self._get_pool()
        if not pool:
            raise Exception("Database connection not established.")
        query = "DELETE FROM email_otps WHERE email = $1;"
        async with pool.acquire() as conn:
            await conn.execute(query, email)

    async def close(self):
        if self.pool:
            await self.pool.close()
            print("[AuthRepository] Supabase Auth DB pool closed. ✅")
