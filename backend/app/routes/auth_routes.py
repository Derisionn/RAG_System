from fastapi import APIRouter, HTTPException, status, Depends, Response, Request
from datetime import datetime, timedelta, timezone
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from ..repositories.auth_repository import AuthRepository
from ..auth.security import get_password_hash, verify_password, create_access_token, generate_refresh_token, hash_refresh_token, verify_token
from ..auth.dependencies import get_current_user
from ..schemas.auth_schemas import (
    UserRegisterPayload, 
    UserLoginPayload, 
    GoogleAuthPayload,
    SetPasswordPayload,
    CheckEmailPayload,
    VerifyOtpPayload
)
from ..config.config import REFRESH_TOKEN_EXPIRE_MINUTES, GOOGLE_CLIENT_ID
from ..utils.email_utils import send_otp_email
import random

router = APIRouter(prefix="/auth", tags=["auth"])

# In a real app you'd use a dependency injection container, but we'll instantiate it here for simplicity
auth_repo = AuthRepository()

# ── Helper ─────────────────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, token: str, remember_me: bool):
    """
    Set the HttpOnly refresh token cookie.
    If remember_me is True → persistent cookie (7-day max_age).
    If remember_me is False → session cookie (expires when browser closes).
    """
    kwargs = dict(
        key="refresh_token",
        value=token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
    )
    if remember_me:
        kwargs["max_age"] = REFRESH_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(**kwargs)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/send-otp")
async def send_otp(payload: CheckEmailPayload):
    """Generates and sends an OTP to the email."""
    # Check if user already exists
    user = await auth_repo.get_user_by_email(payload.email)
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    # Save to DB
    await auth_repo.create_or_update_otp(payload.email, otp, expires_at)
    
    # Send Email
    success = send_otp_email(payload.email, otp)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send verification email")
        
    return {"message": "OTP sent successfully"}

@router.post("/verify-otp")
async def verify_otp(payload: VerifyOtpPayload):
    """Verifies the OTP."""
    otp_record = await auth_repo.get_otp(payload.email)
    
    if not otp_record:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
        
    # Check expiration
    # Ensure both are timezone aware for comparison
    expires_at = otp_record['expires_at']
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
        
    if otp_record['otp'] != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
        
    # Mark as verified
    await auth_repo.mark_otp_verified(payload.email)
    return {"message": "Email verified successfully"}

@router.post("/check-email")
async def check_email(payload: CheckEmailPayload):
    """
    Checks if an email exists and determines its status:
    - 'available': Not registered
    - 'registered': Registered with a password
    - 'google_only': Registered via Google but no password set
    """
    user = await auth_repo.get_user_by_email(payload.email)
    if not user:
        return {"status": "available"}
    if user.hashed_password:
        return {"status": "registered"}
    return {"status": "google_only"}


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserRegisterPayload, request: Request, response: Response):
    # Check if user already exists
    existing = await auth_repo.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Check if email is verified
    otp_record = await auth_repo.get_otp(payload.email)
    if not otp_record or not otp_record['is_verified']:
        raise HTTPException(status_code=400, detail="Email is not verified. Please verify your email first.")
        
    hashed_pw = get_password_hash(payload.password)
    user = await auth_repo.create_user(payload.email, hashed_pw, payload.display_name)
    
    # Clean up OTP record
    await auth_repo.delete_otp(payload.email)
    
    if not user:
        raise HTTPException(status_code=400, detail="Could not create user")
        
    # Return access token in body, refresh token in cookie
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email, "display_name": user.display_name})
    refresh_token = generate_refresh_token()
    rt_hash = hash_refresh_token(refresh_token)
    
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    await auth_repo.create_session(
        user_id=user.id, 
        refresh_token_hash=rt_hash, 
        expires_at=expires_at, 
        ip_address=ip_address, 
        user_agent=user_agent
    )
    
    # New registrations always get a persistent cookie (they just signed up)
    _set_refresh_cookie(response, refresh_token, remember_me=True)
    return {"access_token": access_token, "token_type": "bearer", "display_name": user.display_name}


@router.post("/login")
async def login(payload: UserLoginPayload, request: Request, response: Response):
    user = await auth_repo.get_user_by_email(payload.email)
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email, "display_name": user.display_name})
    refresh_token = generate_refresh_token()
    rt_hash = hash_refresh_token(refresh_token)
    
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    await auth_repo.create_session(
        user_id=user.id, 
        refresh_token_hash=rt_hash, 
        expires_at=expires_at, 
        ip_address=ip_address, 
        user_agent=user_agent
    )
    
    _set_refresh_cookie(response, refresh_token, remember_me=payload.remember_me)
    return {"access_token": access_token, "token_type": "bearer", "display_name": user.display_name}


@router.post("/refresh")
async def refresh_token(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token missing")
        
    rt_hash = hash_refresh_token(token)
    session = await auth_repo.get_session_by_hash(rt_hash)
    
    # Check if session exists and is valid
    if not session or session.revoked_at:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Asyncpg returns aware datetimes. Compare properly.
    now = datetime.now(timezone.utc)
    exp = session.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
        
    if exp < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = await auth_repo.get_user_by_id(session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
        
    # Revoke old session and create a new one (Token Rotation)
    await auth_repo.revoke_session(rt_hash)
    
    new_refresh_token = generate_refresh_token()
    new_rt_hash = hash_refresh_token(new_refresh_token)
    
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    await auth_repo.create_session(
        user_id=session.user_id,
        refresh_token_hash=new_rt_hash,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    access_token = create_access_token(data={"sub": str(session.user_id), "email": user.email, "display_name": user.display_name})
    
    # Preserve persistent cookie on refresh
    _set_refresh_cookie(response, new_refresh_token, remember_me=True)
    needs_password = True if user.hashed_password is None else False
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "display_name": user.display_name,
        "needs_password": needs_password
    }


@router.post("/google")
async def google_auth(payload: GoogleAuthPayload, request: Request, response: Response):
    """
    Verify a Google id_token from the frontend, then find-or-create a user.
    - If google_id already in DB  → log in directly
    - If email already in DB      → link google_id, then log in
    - Neither                     → create new OAuth user, then log in
    """
    # 1. Verify the token with Google's public keys
    try:
        idinfo = id_token.verify_oauth2_token(
            payload.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Google token: {str(e)}")

    google_id   = idinfo["sub"]
    email       = idinfo.get("email", "")
    name        = idinfo.get("name") or idinfo.get("given_name") or email.split("@")[0]

    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email address")

    # 2. Find or create user
    user = await auth_repo.get_user_by_google_id(google_id)

    if not user:
        # Check if an email/password account already exists with this email
        user = await auth_repo.get_user_by_email(email)
        if user:
            # Auto-link: attach google_id to the existing account
            await auth_repo.link_google_id(user.id, google_id)
            print(f"[GoogleAuth] Linked google_id to existing account: {email}")
        else:
            # Brand-new user — create without a password
            user = await auth_repo.create_oauth_user(email, name, google_id)
            if not user:
                raise HTTPException(status_code=500, detail="Failed to create user account")
            print(f"[GoogleAuth] Created new OAuth user: {email}")
    else:
        print(f"[GoogleAuth] Existing Google user logged in: {email}")

    # 3. Issue JWT + refresh token (same flow as email login)
    access_token  = create_access_token(data={"sub": str(user.id), "email": user.email, "display_name": user.display_name})
    refresh_token = generate_refresh_token()
    rt_hash       = hash_refresh_token(refresh_token)

    expires_at  = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    ip_address  = request.client.host if request.client else None
    user_agent  = request.headers.get("user-agent")

    await auth_repo.create_session(
        user_id=user.id,
        refresh_token_hash=rt_hash,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )

    # Google logins always get a persistent cookie
    _set_refresh_cookie(response, refresh_token, remember_me=True)
    needs_password = True if user.hashed_password is None else False
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "display_name": user.display_name,
        "needs_password": needs_password
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if token:
        rt_hash = hash_refresh_token(token)
        await auth_repo.revoke_session(rt_hash)
    response.delete_cookie("refresh_token")
    return {"message": "Logged out successfully"}

@router.post("/set-password")
async def set_password(payload: SetPasswordPayload, current_user = Depends(get_current_user)):
    """
    Allows a Google SSO user to set a password for their account.
    """
    user_id = int(current_user["id"])
    user = await auth_repo.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user.hashed_password:
        raise HTTPException(status_code=400, detail="User already has a password set")
    
    hashed = get_password_hash(payload.password)
    await auth_repo.update_password(user_id, hashed)
    
    return {"message": "Password set successfully"}
