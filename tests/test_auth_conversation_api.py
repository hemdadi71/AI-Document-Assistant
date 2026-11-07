import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main as main_module
from core.auth_dependencies import AuthContext, get_current_user
from core.auth_service import (
    InvalidCredentials,
    InvalidRefreshToken,
    UsernameAlreadyExists,
)
from database.session import get_db
from core.auth_cookies import (
    ACCESS_COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    CSRF_COOKIE_NAME,
)


class FakeDB:
    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture
def client():
    def override_db():
        yield FakeDB()

    main_module.app.dependency_overrides[get_db] = override_db
    with TestClient(main_module.app) as test_client:
        yield test_client
    main_module.app.dependency_overrides.clear()


def test_register_login_and_refresh_endpoints(client, monkeypatch):
    user = SimpleNamespace(
        id=uuid.uuid4(),
        username="test_user",
    )

    class FakeAuthService:
        def __init__(self, db):
            self.db = db

        def register(self, username, password):
            assert username == "test_user"
            assert password == "password123"
            return user

        def login(self, username, password, project_id):
            assert project_id == "chat_only_demo"
            return {
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "bearer",
                "expires_in": 900,
                "user": user,
            }

        def refresh(self, refresh_token):
            assert refresh_token == "refresh"
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "token_type": "bearer",
                "expires_in": 900,
            }

    monkeypatch.setattr(main_module, "AuthService", FakeAuthService)

    registered = client.post(
        "/auth/register",
        json={
            "username": "Test_User",
            "password": "password123",
            "project_id": "chat_only_demo",
        },
    )
    assert registered.status_code == 201
    assert registered.json()["user"]["username"] == "test_user"
    assert "access_token" not in registered.json()
    assert ACCESS_COOKIE_NAME in registered.cookies
    assert REFRESH_COOKIE_NAME in registered.cookies
    assert CSRF_COOKIE_NAME in registered.cookies

    # Registration also signs the browser in. Subsequent unsafe requests
    # must echo the readable CSRF cookie in the request header.
    csrf = client.cookies.get(CSRF_COOKIE_NAME)
    logged_in = client.post(
        "/auth/login",
        headers={"X-CSRF-Token": csrf},
        json={
            "username": "test_user",
            "password": "password123",
            "project_id": "chat_only_demo",
        },
    )
    assert logged_in.status_code == 200
    assert "access_token" not in logged_in.json()
    assert logged_in.json()["user"]["id"] == str(user.id)
    assert logged_in.cookies.get(ACCESS_COOKIE_NAME) == "access"

    csrf = client.cookies.get(CSRF_COOKIE_NAME)
    refreshed = client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": csrf},
    )
    assert refreshed.status_code == 200
    assert "access_token" not in refreshed.json()
    assert client.cookies.get(ACCESS_COOKIE_NAME) == "new-access"


def test_register_duplicate_username_returns_conflict(client, monkeypatch):
    class FakeAuthService:
        def __init__(self, db):
            pass

        def register(self, username, password):
            raise UsernameAlreadyExists()

    monkeypatch.setattr(main_module, "AuthService", FakeAuthService)

    response = client.post(
        "/auth/register",
        json={"username": "taken", "password": "password123"},
    )
    assert response.status_code == 409


def test_login_invalid_credentials_returns_401(client, monkeypatch):
    class FakeAuthService:
        def __init__(self, db):
            pass

        def login(self, username, password, project_id):
            raise InvalidCredentials()

    monkeypatch.setattr(main_module, "AuthService", FakeAuthService)

    response = client.post(
        "/auth/login",
        json={"username": "unknown", "password": "wrong"},
    )
    assert response.status_code == 401


def test_invalid_refresh_token_returns_401(client, monkeypatch):
    class FakeAuthService:
        def __init__(self, db):
            pass

        def refresh(self, refresh_token):
            raise InvalidRefreshToken()

    monkeypatch.setattr(main_module, "AuthService", FakeAuthService)

    client.cookies.set(REFRESH_COOKIE_NAME, "invalid", path="/auth")
    client.cookies.set(CSRF_COOKIE_NAME, "csrf-test", path="/")
    response = client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": "csrf-test"},
    )
    assert response.status_code == 401


def test_conversation_endpoints_are_authenticated_and_scoped(client, monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4(), username="owner")
    auth = AuthContext(
        user=user,
        project_id="chat_only_demo",
        session_id=uuid.uuid4(),
    )
    main_module.app.dependency_overrides[get_current_user] = lambda: auth
    conversation_id = uuid.uuid4()

    class FakeConversationService:
        def __init__(self, db):
            pass

        def list_conversations(self, user_id, project_id, limit):
            assert user_id == user.id
            assert project_id == auth.project_id
            return [{"id": str(conversation_id), "title": "A chat"}]

        def get_conversation(self, conversation_id, user_id, project_id):
            assert user_id == user.id
            assert project_id == auth.project_id
            return {"id": str(conversation_id), "messages": []}

        def delete_conversation(self, conversation_id, user_id, project_id):
            assert user_id == user.id
            assert project_id == auth.project_id
            return True

    monkeypatch.setattr(
        main_module,
        "ConversationService",
        FakeConversationService,
    )

    listed = client.get("/conversations")
    loaded = client.get(f"/conversations/{conversation_id}")
    deleted = client.delete(f"/conversations/{conversation_id}")

    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert loaded.status_code == 200
    assert loaded.json()["id"] == str(conversation_id)
    assert deleted.status_code == 200


def test_jwt_chat_restores_history_and_persists_new_turn(client, monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4(), username="owner")
    auth = AuthContext(
        user=user,
        project_id="chat_only_demo",
        session_id=uuid.uuid4(),
    )
    principal = main_module.RequestPrincipal(
        project_id=auth.project_id,
        auth_context=auth,
    )
    main_module.app.dependency_overrides[
        main_module.get_request_principal
    ] = lambda: principal

    conversation = SimpleNamespace(id=uuid.uuid4())
    captured = {}

    class FakeDBSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def commit(self):
            pass

        def rollback(self):
            pass

    monkeypatch.setattr(
        main_module,
        "get_session_factory",
        lambda: FakeDBSession,
    )

    class FakeConversationService:
        def __init__(self, db):
            pass

        def get_or_create(self, **kwargs):
            assert kwargs["user_id"] == user.id
            assert kwargs["project_id"] == auth.project_id
            return conversation

        def get_history(self, conversation_id):
            assert conversation_id == conversation.id
            return [{"role": "user", "content": "Earlier"}]

        def save_turn(self, **kwargs):
            captured["saved"] = kwargs

    class FakeRuntime:
        config = SimpleNamespace(
            name="chat_only_demo",
            chat_enabled=True,
            rag_enabled=True,
        )
        rag_service = object()
        system_prompt = "system"

        def ask(self, question, **kwargs):
            captured["history"] = kwargs["chat_service"].history
            captured["conversation_id"] = kwargs["conversation_id"]
            return {"answer": "A reply", "sources": []}

    monkeypatch.setattr(
        main_module,
        "ConversationService",
        FakeConversationService,
    )
    monkeypatch.setattr(
        main_module,
        "get_authorized_runtime",
        lambda *args: FakeRuntime(),
    )

    response = client.post(
        "/chat",
        json={"question": "Follow-up", "mode": "chat"},
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == str(conversation.id)
    assert captured["history"] == [
        {"role": "user", "content": "Earlier"}
    ]
    assert captured["saved"]["answer"] == "A reply"


def test_jwt_chat_stream_persists_completed_response(client, monkeypatch):
    user = SimpleNamespace(id=uuid.uuid4(), username="owner")
    auth = AuthContext(
        user=user,
        project_id="general_assistant",
        session_id=uuid.uuid4(),
    )
    principal = main_module.RequestPrincipal(
        project_id=auth.project_id,
        auth_context=auth,
    )
    main_module.app.dependency_overrides[
        main_module.get_request_principal
    ] = lambda: principal

    conversation = SimpleNamespace(id=uuid.uuid4())
    captured = {}

    class FakeDBSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def commit(self):
            pass

        def rollback(self):
            pass

        def get(self, model, key):
            assert key == conversation.id
            return conversation

    monkeypatch.setattr(
        main_module,
        "get_session_factory",
        lambda: FakeDBSession,
    )

    class FakeConversationService:
        def __init__(self, db):
            pass

        def get_or_create(self, **kwargs):
            assert kwargs["user_id"] == user.id
            return conversation

        def get_history(self, conversation_id):
            return []

        def save_turn(self, **kwargs):
            captured["saved"] = kwargs

    class FakeRuntime:
        config = SimpleNamespace(
            name="general_assistant",
            chat_enabled=False,
            rag_enabled=True,
        )
        rag_service = object()

        def stream_answer(self, question, history=None):
            assert question == "Question"
            return iter(["Part one", " part two"])

    monkeypatch.setattr(
        main_module,
        "ConversationService",
        FakeConversationService,
    )
    monkeypatch.setattr(
        main_module,
        "get_authorized_runtime",
        lambda *args: FakeRuntime(),
    )

    response = client.post(
        "/chat-stream",
        json={"question": "Question", "mode": "rag"},
    )

    assert response.status_code == 200
    assert response.headers["x-conversation-id"] == str(conversation.id)
    assert response.text == "Part one part two"
    assert captured["saved"]["answer"] == "Part one part two"


def test_cookie_authenticated_write_requires_csrf_header(client):
    client.cookies.set(ACCESS_COOKIE_NAME, "access-test", path="/")
    client.cookies.set(CSRF_COOKIE_NAME, "expected-csrf", path="/")

    missing_header = client.post("/auth/refresh")
    assert missing_header.status_code == 403

    wrong_header = client.post(
        "/auth/refresh",
        headers={"X-CSRF-Token": "wrong-csrf"},
    )
    assert wrong_header.status_code == 403


def test_logout_revokes_session_and_clears_auth_cookies(
    client,
    monkeypatch,
):
    user = SimpleNamespace(id=uuid.uuid4(), username="owner")
    auth = AuthContext(
        user=user,
        project_id="chat_only_demo",
        session_id=uuid.uuid4(),
    )
    captured = {}

    class FakeAuthService:
        def __init__(self, db):
            pass

        def logout(self, session_id):
            captured["session_id"] = session_id
            return True

        def logout_refresh_token(self, refresh_token):
            captured["refresh_token"] = refresh_token
            return True

    monkeypatch.setattr(main_module, "AuthService", FakeAuthService)
    monkeypatch.setattr(
        main_module,
        "resolve_access_token",
        lambda token, db: auth,
    )

    client.cookies.set(ACCESS_COOKIE_NAME, "access-test", path="/")
    client.cookies.set(REFRESH_COOKIE_NAME, "refresh-test", path="/auth")
    client.cookies.set(CSRF_COOKIE_NAME, "csrf-test", path="/")

    response = client.post(
        "/auth/logout",
        headers={"X-CSRF-Token": "csrf-test"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "logged_out"
    assert captured["session_id"] == auth.session_id
    assert client.cookies.get(ACCESS_COOKIE_NAME) is None
    assert client.cookies.get(REFRESH_COOKIE_NAME) is None
    assert client.cookies.get(CSRF_COOKIE_NAME) is None
