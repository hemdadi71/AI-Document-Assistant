import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import core.auth_dependencies as auth_dependencies


def make_auth_fixture(project_id="chat_only_demo"):
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, is_active=True)
    session = SimpleNamespace(
        id=session_id,
        user_id=user_id,
        project_id=project_id,
        revoked_at=None,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    db = SimpleNamespace(
        get=lambda model, key: (
            session if model.__name__ == "UserSession" else user
        )
    )
    claims = {
        "sub": str(user_id),
        "sid": str(session_id),
        "project_id": project_id,
    }
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="valid-token",
    )
    return user, session, db, claims, credentials


def test_resolve_current_user_validates_database_session(monkeypatch):
    user, session, db, claims, credentials = make_auth_fixture()
    monkeypatch.setattr(
        auth_dependencies,
        "decode_token",
        lambda token: claims,
    )

    result = auth_dependencies.resolve_current_user(credentials, db)

    assert result.user is user
    assert result.project_id == "chat_only_demo"
    assert result.session_id == session.id


def test_resolve_current_user_rejects_project_mismatch(monkeypatch):
    user, session, db, claims, credentials = make_auth_fixture()
    session.project_id = "another-project"
    monkeypatch.setattr(
        auth_dependencies,
        "decode_token",
        lambda token: claims,
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_dependencies.resolve_current_user(credentials, db)

    assert exc_info.value.status_code == 401


def test_resolve_current_user_rejects_revoked_session(monkeypatch):
    user, session, db, claims, credentials = make_auth_fixture()
    session.revoked_at = datetime.now(timezone.utc)
    monkeypatch.setattr(
        auth_dependencies,
        "decode_token",
        lambda token: claims,
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_dependencies.resolve_current_user(credentials, db)

    assert exc_info.value.status_code == 401



def test_get_current_user_accepts_access_cookie(monkeypatch):
    user, session, db, claims, credentials = make_auth_fixture()
    monkeypatch.setattr(
        auth_dependencies,
        "decode_token",
        lambda token: claims,
    )

    result = auth_dependencies.get_current_user(
        credentials=None,
        access_token="valid-cookie-token",
        db=db,
    )

    assert result.user is user
    assert result.project_id == "chat_only_demo"
