from fastapi import APIRouter, HTTPException, status, Depends, Response, Request
from datetime import datetime, timedelta, timezone
from ..repositories.auth_repository import AuthRepository
from ..auth.security import get_password_hash, verify_password, create_access_token, generate_refresh_token, hash_refresh_token, verify_token
from ..schemas.auth_schemas import UserCredentials
from ..config.config import REFRESH_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["auth"])

# In a real app you'd use a dependency injection container, but we'll instantiate it here for simplicity
auth_repo = AuthRepository()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(creds: UserCredentials, request: Request, response: Response):
    # Check if user already exists
    existing = await auth_repo.get_user_by_email(creds.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_pw = get_password_hash(creds.password)
    user = await auth_repo.create_user(creds.email, hashed_pw)
    
    if not user:
        raise HTTPException(status_code=400, detail="Could not create user")
        
    # Return access token in body, refresh token in cookie
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
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
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False, # Set to True in production with HTTPS
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login")
async def login(creds: UserCredentials, request: Request, response: Response):
    user = await auth_repo.get_user_by_email(creds.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not verify_password(creds.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
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
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=False, # Set to True in production
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60
    )
    return {"access_token": access_token, "token_type": "bearer"}

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
    # Ensure session.expires_at is timezone aware if not already
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
        
    user_email = user.email
    
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
    
    access_token = create_access_token(data={"sub": str(session.user_id), "email": user_email})
    
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if token:
        rt_hash = hash_refresh_token(token)
        await auth_repo.revoke_session(rt_hash)
    response.delete_cookie("refresh_token")
    return {"message": "Logged out successfully"}
