
import os
import uuid
import hashlib
import secrets

from pathlib import Path
from datetime import datetime, timedelta, timezone

import jwt

from dotenv import load_dotenv
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import User, UserSession


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env.docker", override=False)

JWT_ALGORITHM = "HS256"

password_hasher = PasswordHash.recommended()


DUMMY_PASSWORD_HASH = password_hasher.hash(
    "dummy-password-for-timing-protection"
)


class AuthError(Exception):
    pass


class UsernameAlreadyExists(AuthError):
    pass


class InvalidCredentials(AuthError):
    pass


class InvalidRefreshToken(AuthError):
    pass


def get_jwt_secret() -> str:
    secret = os.getenv("AUTH_SECRET_KEY", "").strip()

    if len(secret) < 32:
        raise RuntimeError(
            "AUTH_SECRET_KEY must contain at least 32 characters."
        )

    return secret


def get_access_minutes() -> int:
    return int(
        os.getenv("AUTH_ACCESS_TOKEN_MINUTES", "15")
    )


def get_refresh_days() -> int:
    return int(
        os.getenv("AUTH_REFRESH_TOKEN_DAYS", "30")
    )


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    try:
        return password_hasher.verify(
            password,
            hashed_password,
        )
    except Exception:
        return False


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def decode_token(token: str) -> dict:

    try:
        claims = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            options={
                "require": [
                    "sub",
                    "exp",
                    "iat",
                    "type",
                    "sid",
                    "project_id",
                ]
            },
        )

        if claims.get("type") != "access":
            raise InvalidCredentials()

        uuid.UUID(claims["sub"])
        uuid.UUID(claims["sid"])

        if not isinstance(claims["project_id"], str):
            raise InvalidCredentials()

        if not claims["project_id"].strip():
            raise InvalidCredentials()

        return claims

    except (
        jwt.InvalidTokenError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        raise InvalidCredentials(
            "Invalid or expired access token."
        ) from exc


class AuthService:

    def __init__(self, db: Session):
        self.db = db

    def register(
        self,
        username: str,
        password: str,
    ) -> User:

        username = username.strip().lower()

        existing = self.db.scalar(
            select(User).where(
                User.username == username
            )
        )

        if existing:
            raise UsernameAlreadyExists(
                "Username is already registered."
            )

        user = User(
            username=username,
            password_hash=hash_password(password),
            is_active=True,
        )

        self.db.add(user)

        try:
            self.db.commit()
            self.db.refresh(user)

        except IntegrityError as exc:
            self.db.rollback()

            raise UsernameAlreadyExists(
                "Username is already registered."
            ) from exc

        return user

    def _create_session(
        self,
        user: User,
        project_id: str,
    ) -> dict:

        now = datetime.now(timezone.utc)

        refresh_expires = now + timedelta(
            days=get_refresh_days()
        )

        access_expires = now + timedelta(
            minutes=get_access_minutes()
        )

        refresh_token = secrets.token_urlsafe(48)

        session = UserSession(
            user_id=user.id,
            project_id=project_id,
            token_hash=hash_refresh_token(refresh_token),
            expires_at=refresh_expires,
        )

        self.db.add(session)
        self.db.flush()

        access_token = jwt.encode(
            {
                "sub": str(user.id),
                "sid": str(session.id),
                "project_id": project_id,
                "type": "access",
                "iat": int(now.timestamp()),
                "exp": int(access_expires.timestamp()),
            },
            get_jwt_secret(),
            algorithm=JWT_ALGORITHM,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": get_access_minutes() * 60,
        }

    def login(
        self,
        username: str,
        password: str,
        project_id: str,
    ) -> dict:

        username = username.strip().lower()

        user = self.db.scalar(
            select(User).where(
                User.username == username
            )
        )

        if user is None:
            verify_password(
                password,
                DUMMY_PASSWORD_HASH,
            )

            raise InvalidCredentials(
                "Invalid username or password."
            )

        if not verify_password(
            password,
            user.password_hash,
        ):
            raise InvalidCredentials(
                "Invalid username or password."
            )

        if not user.is_active:
            raise InvalidCredentials(
                "Invalid username or password."
            )

        tokens = self._create_session(
            user,
            project_id,
        )

        self.db.commit()

        return {
            **tokens,
            "user": user,
        }

    def refresh(
        self,
        refresh_token: str,
    ) -> dict:
        """
        Rotate a refresh token and preserve its original project binding.

        Only the hash of the refresh token is stored in the database.
        The project ID is read from the locked session, never from
        untrusted request data.
        """

        token_hash = hash_refresh_token(refresh_token)

        # Lock the session row to prevent concurrent token rotation.
        session = self.db.scalar(
            select(UserSession)
            .where(UserSession.token_hash == token_hash)
            .with_for_update()
        )

        if session is None:
            self.db.rollback()
            raise InvalidRefreshToken("Invalid refresh token.")

        now = datetime.now(timezone.utc)
        expires_at = session.expires_at

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if session.revoked_at is not None or expires_at <= now:
            self.db.rollback()
            raise InvalidRefreshToken(
                "Refresh token is expired or revoked."
            )

        user = self.db.get(User, session.user_id)

        if user is None or not user.is_active:
            session.revoked_at = now
            self.db.commit()
            raise InvalidRefreshToken("User is unavailable.")

        # Preserve the project bound to the original session.
        project_id = session.project_id

        # Revoke the old session before issuing a new refresh token.
        session.revoked_at = now

        tokens = self._create_session(
            user,
            project_id,
        )

        self.db.commit()

        return tokens

    def logout_refresh_token(
        self,
        refresh_token: str,
    ) -> bool:
        """Revoke the session identified by a refresh token, if it exists."""
        token_hash = hash_refresh_token(refresh_token)

        session = self.db.scalar(
            select(UserSession)
            .where(UserSession.token_hash == token_hash)
            .with_for_update()
        )

        if session is None:
            self.db.rollback()
            return False

        if session.revoked_at is None:
            session.revoked_at = datetime.now(timezone.utc)

        self.db.commit()
        return True

    def logout(
        self,
        session_id: uuid.UUID,
    ) -> bool:

        session = self.db.scalar(
            select(UserSession)
            .where(
                UserSession.id == session_id
            )
            .with_for_update()
        )

        if session is None:
            self.db.rollback()
            return False

        if session.revoked_at is None:
            session.revoked_at = datetime.now(
                timezone.utc
            )

        self.db.commit()

        return True
