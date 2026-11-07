import uuid

from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Cookie, Depends, HTTPException
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from sqlalchemy.orm import Session

from database.session import get_db
from database.models import User, UserSession

from core.auth_service import decode_token, InvalidCredentials
from core.auth_cookies import ACCESS_COOKIE_NAME


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    user: User
    project_id: str
    session_id: uuid.UUID


def resolve_access_token(
    access_token: str,
    db: Session,
) -> AuthContext:
    """Validate an access token and its persisted database session."""

    try:
        claims = decode_token(access_token)
        user_id = uuid.UUID(claims["sub"])
        session_id = uuid.UUID(claims["sid"])
        project_id = claims["project_id"]

    except (
        InvalidCredentials,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    session = db.get(UserSession, session_id)
    user = db.get(User, user_id)

    if session is None or user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid session.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    now = datetime.now(timezone.utc)
    expires_at = session.expires_at

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if (
        session.user_id != user_id
        or session.project_id != project_id
        or session.revoked_at is not None
        or expires_at <= now
        or not user.is_active
    ):
        raise HTTPException(
            status_code=401,
            detail="Session is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthContext(
        user=user,
        project_id=project_id,
        session_id=session_id,
    )


def resolve_current_user(
    credentials: HTTPAuthorizationCredentials,
    db: Session,
) -> AuthContext:
    """Backward-compatible validation for Authorization: Bearer clients."""
    return resolve_access_token(credentials.credentials, db)


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    access_token: str | None = Cookie(
        default=None,
        alias=ACCESS_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
) -> AuthContext | None:
    """Return None without credentials; accept Bearer or browser cookie."""

    token = (
        credentials.credentials
        if credentials is not None
        else access_token
    )
    if token is None:
        return None

    return resolve_access_token(token, db)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    access_token: str | None = Cookie(
        default=None,
        alias=ACCESS_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
) -> AuthContext:
    """Require a valid Bearer token or HttpOnly access cookie."""

    token = (
        credentials.credentials
        if credentials is not None
        else access_token
    )

    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return resolve_access_token(token, db)
