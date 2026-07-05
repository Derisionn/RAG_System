from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import re


class CheckEmailPayload(BaseModel):
    """Payload to check if an email exists."""
    email: EmailStr

class VerifyOtpPayload(BaseModel):
    """Payload to verify an OTP."""
    email: EmailStr
    otp: str

class UserRegisterPayload(BaseModel):
    """Payload for new user registration — includes optional display name."""
    email: EmailStr
    password: str
    display_name: Optional[str] = None

    @field_validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r"[@$!%*?&]", v):
            raise ValueError("Password must contain at least one special character (@$!%*?&)")
        return v

    @field_validator("display_name")
    def validate_display_name(cls, v):
        if v is not None:
            v = v.strip()
            if len(v) < 2:
                raise ValueError("Display name must be at least 2 characters")
            if len(v) > 50:
                raise ValueError("Display name must be 50 characters or fewer")
        return v


class UserLoginPayload(BaseModel):
    """Payload for login — includes remember_me to control cookie lifetime."""
    email: EmailStr
    password: str
    remember_me: bool = False


class GoogleAuthPayload(BaseModel):
    """Payload received from the frontend's Google Login button."""
    credential: str

class SetPasswordPayload(BaseModel):
    """Payload for setting a password after Google SSO."""
    password: str

    @field_validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r"[@$!%*?&]", v):
            raise ValueError("Password must contain at least one special character (@$!%*?&)")
        return v


# Keep UserCredentials as a legacy alias so existing imports don't break
class UserCredentials(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str
