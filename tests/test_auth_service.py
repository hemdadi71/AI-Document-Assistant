from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.auth_service import AuthService, InvalidRefreshToken


class FakeDB:
    def __init__(self, session, user):
        self.session = session
        self.user = user
        self.committed = False
        self.rolled_back = False

    def scalar(self, statement):
        return self.session

    def get(self, model, key):
        return self.user

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_refresh_preserves_project_from_persisted_session(monkeypatch):
    user_id = "user-id"
    user = SimpleNamespace(id=user_id, is_active=True)
    session = SimpleNamespace(
        user_id=user_id,
        project_id="original-project",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        revoked_at=None,
    )
    db = FakeDB(session, user)
    service = AuthService(db)
    captured = {}

    def fake_create_session(current_user, project_id):
        captured["user"] = current_user
        captured["project_id"] = project_id
        return {"access_token": "new-access"}

    monkeypatch.setattr(service, "_create_session", fake_create_session)

    result = service.refresh("old-refresh-token")

    assert result == {"access_token": "new-access"}
    assert captured["user"] is user
    assert captured["project_id"] == "original-project"
    assert session.revoked_at is not None
    assert db.committed is True


def test_refresh_rejects_revoked_session(monkeypatch):
    user = SimpleNamespace(id="user-id", is_active=True)
    session = SimpleNamespace(
        user_id="user-id",
        project_id="original-project",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        revoked_at=datetime.now(timezone.utc),
    )
    db = FakeDB(session, user)
    service = AuthService(db)

    monkeypatch.setattr(
        service,
        "_create_session",
        lambda *args: pytest.fail("A revoked session must not be rotated."),
    )

    with pytest.raises(InvalidRefreshToken):
        service.refresh("revoked-refresh-token")

    assert db.rolled_back is True



def test_logout_refresh_token_revokes_session():
    user = SimpleNamespace(id="user-id", is_active=True)
    session = SimpleNamespace(
        revoked_at=None,
    )
    db = FakeDB(session, user)
    service = AuthService(db)

    result = service.logout_refresh_token("refresh-token")

    assert result is True
    assert session.revoked_at is not None
    assert db.committed is True
