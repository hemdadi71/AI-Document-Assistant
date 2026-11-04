
import pytest
from fastapi.testclient import TestClient

import main as main_module

from core.project_config import ProjectConfig
from core.project_registry import ProjectRegistry
from core.security import api_key_env_name


PROJECT_A_KEY = "a" * 40
PROJECT_B_KEY = "b" * 40


class FakeRuntime:
    """Minimal runtime for API isolation tests."""

    def __init__(self, config):
        self.config = config
        self.rag_service = (
            object() if config.rag_enabled else None
        )

    def list_documents(self):
        return []

    def ask(
        self,
        question,
        mode="rag",
        conversation_id=None,
    ):
        return {
            "answer": f"Answer: {question}",
            "sources": [],
            "conversation_id": conversation_id,
        }


@pytest.fixture
def api_client(monkeypatch):
    project_a = ProjectConfig(
        name="project_a",
        chat_enabled=False,
        rag_enabled=True,
    )

    project_b = ProjectConfig(
        name="project_b",
        chat_enabled=True,
        rag_enabled=False,
    )

    # Configure independent API keys for the test projects.
    monkeypatch.setenv(
        api_key_env_name("project_a"),
        PROJECT_A_KEY,
    )
    monkeypatch.setenv(
        api_key_env_name("project_b"),
        PROJECT_B_KEY,
    )

    runtime_a = FakeRuntime(project_a)
    runtime_b = FakeRuntime(project_b)

    monkeypatch.setattr(
        main_module,
        "project_registry",
        ProjectRegistry([project_a, project_b]),
    )

    monkeypatch.setattr(
        main_module,
        "project_runtimes",
        {
            "project_a": runtime_a,
            "project_b": runtime_b,
        },
    )

    return TestClient(main_module.app)


def test_missing_api_key_returns_401(api_client):
    response = api_client.get("/health")

    assert response.status_code == 401


def test_invalid_api_key_returns_401(api_client):
    response = api_client.get(
        "/health",
        headers={"X-API-Key": "x" * 40},
    )

    assert response.status_code == 401


def test_project_can_access_its_own_health(api_client):
    response = api_client.get(
        "/health",
        headers={"X-API-Key": PROJECT_A_KEY},
    )

    assert response.status_code == 200
    assert response.json()["project"] == "project_a"
    assert response.json()["features"] == {
        "chat": False,
        "rag": True,
        "hybrid": False,
    }


def test_project_cannot_access_another_project(api_client):
    response = api_client.get(
        "/health?project_id=project_b",
        headers={"X-API-Key": PROJECT_A_KEY},
    )

    assert response.status_code == 403


def test_chat_only_project_cannot_access_rag_documents(
    api_client,
):
    response = api_client.get(
        "/documents",
        headers={"X-API-Key": PROJECT_B_KEY},
    )

    assert response.status_code == 403


def test_rag_only_project_cannot_use_chat_mode(api_client):
    response = api_client.post(
        "/chat",
        headers={"X-API-Key": PROJECT_A_KEY},
        json={
            "question": "Hello",
            "mode": "chat",
        },
    )

    assert response.status_code == 403


def test_chat_only_project_cannot_use_rag_mode(api_client):
    response = api_client.post(
        "/chat",
        headers={"X-API-Key": PROJECT_B_KEY},
        json={
            "question": "Search my documents",
            "mode": "rag",
        },
    )

    assert response.status_code == 403


def test_hybrid_requires_both_features(api_client):
    for api_key in (PROJECT_A_KEY, PROJECT_B_KEY):
        response = api_client.post(
            "/chat",
            headers={"X-API-Key": api_key},
            json={
                "question": "Test hybrid",
                "mode": "hybrid",
            },
        )

        assert response.status_code == 403


def test_chat_only_project_can_use_chat_mode(api_client):
    response = api_client.post(
        "/chat",
        headers={"X-API-Key": PROJECT_B_KEY},
        json={
            "question": "Hello",
            "mode": "chat",
            "conversation_id": "test-session",
        },
    )

    assert response.status_code == 200
    assert response.json()["project_id"] == "project_b"
    assert response.json()["answer"] == "Answer: Hello"
    assert response.json()["conversation_id"] == "test-session"


def test_chat_request_cannot_override_project_identity(
    api_client,
):
    response = api_client.post(
        "/chat",
        headers={"X-API-Key": PROJECT_A_KEY},
        json={
            "question": "Hello",
            "mode": "rag",
            "project_id": "project_b",
        },
    )

    assert response.status_code == 403
