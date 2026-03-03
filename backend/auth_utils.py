import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional

import bcrypt

if not hasattr(bcrypt, "__about__"):
    version = getattr(bcrypt, "__version__", "0.0.0")
    bcrypt.__about__ = SimpleNamespace(__version__=version)

from jose import JWTError, jwt
from passlib.context import CryptContext

from database import get_admin_setting

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_production")
ALGORITHM = "HS256"

# Default values if not in DB
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        minutes = int(get_admin_setting("access_token_expire_minutes", DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES))
        expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a long-lived JWT refresh token."""
    to_encode = data.copy()
    days = int(get_admin_setting("refresh_token_expire_days", DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS))
    expire = datetime.now(timezone.utc) + timedelta(days=days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
