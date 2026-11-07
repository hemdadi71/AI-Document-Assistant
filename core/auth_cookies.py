"""Secure cookie helpers for browser-based authentication."""

import os
import secrets

from fastapi import Response


ACCESS_COOKIE_NAME = "ai_access_token"
REFRESH_COOKIE_NAME = "ai_refresh_token"
CSRF_COOKIE_NAME = "ai_csrf_token"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cookie_settings() -> dict:
    """Return validated cookie attributes shared by set/delete operations."""
    secure = _env_bool("AUTH_COOKIE_SECURE", default=False)
    samesite = os.getenv("AUTH_COOKIE_SAMESITE", "lax").strip().lower()

    if samesite not in {"lax", "strict", "none"}:
        raise RuntimeError(
            "AUTH_COOKIE_SAMESITE must be lax, strict, or none."
        )

    if samesite == "none" and not secure:
        raise RuntimeError(
            "AUTH_COOKIE_SECURE must be true when SameSite=None."
        )

    domain = os.getenv("AUTH_COOKIE_DOMAIN", "").strip() or None

    return {
        "secure": secure,
        "samesite": samesite,
        "domain": domain,
    }


def set_auth_cookies(response: Response, tokens: dict) -> str:
    """Set HttpOnly access, refresh, and CSRF cookies."""
    from core.auth_service import get_refresh_days

    settings = cookie_settings()
    refresh_max_age = get_refresh_days() * 24 * 60 * 60
    access_max_age = int(tokens["expires_in"])
    csrf_token = secrets.token_urlsafe(32)

    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=tokens["access_token"],
        max_age=access_max_age,
        path="/",
        httponly=True,
        **settings,
    )
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=tokens["refresh_token"],
        max_age=refresh_max_age,
        path="/auth",
        httponly=True,
        **settings,
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=refresh_max_age,
        path="/",
        httponly=True,
        **settings,
    )

    return csrf_token


def clear_auth_cookies(response: Response) -> None:
    """Expire cookies using exactly the same path/domain attributes."""
    settings = cookie_settings()

    response.delete_cookie(
        key=ACCESS_COOKIE_NAME,
        path="/",
        httponly=True,
        **settings,
    )
    response.delete_cookie(
        key=REFRESH_COOKIE_NAME,
        path="/auth",
        httponly=True,
        **settings,
    )
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        path="/",
        httponly=True,
        **settings,
    )
