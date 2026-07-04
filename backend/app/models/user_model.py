from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserInDB(BaseModel):
    """
    Represents the strict internal data model for a User exactly as stored 
    in the Supabase Auth database table.
    """
    id: int
    email: EmailStr
    hashed_password: str
    created_at: datetime

class UserSession(BaseModel):
    """
    Represents an active refresh token session for a user.
    """
    id: str
    user_id: int
    refresh_token_hash: str
    expires_at: datetime
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
