
import pytest

from core.security import (
    APIKeyConfigurationError,
    api_key_env_name,
    resolve_project_for_api_key,
)


def test_resolve_valid_project_key(monkeypatch):
    monkeypatch.setenv(
        api_key_env_name("project_a"),
        "a" * 40,
    )
    monkeypatch.setenv(
        api_key_env_name("project_b"),
        "b" * 40,
    )

    assert resolve_project_for_api_key(
        "a" * 40,
        ["project_a", "project_b"],
    ) == "project_a"


def test_invalid_key_is_rejected(monkeypatch):
    monkeypatch.setenv(
        api_key_env_name("project_a"),
        "a" * 40,
    )

    assert resolve_project_for_api_key(
        "x" * 40,
        ["project_a"],
    ) is None


def test_missing_project_key_raises_error(monkeypatch):
    monkeypatch.delenv(
        api_key_env_name("project_a"),
        raising=False,
    )

    with pytest.raises(APIKeyConfigurationError):
        resolve_project_for_api_key(
            "a" * 40,
            ["project_a"],
        )


def test_duplicate_project_keys_are_rejected(monkeypatch):
    monkeypatch.setenv(
        api_key_env_name("project_a"),
        "a" * 40,
    )
    monkeypatch.setenv(
        api_key_env_name("project_b"),
        "a" * 40,
    )

    with pytest.raises(
        APIKeyConfigurationError,
        match="unique",
    ):
        resolve_project_for_api_key(
            "a" * 40,
            ["project_a", "project_b"],
        )
