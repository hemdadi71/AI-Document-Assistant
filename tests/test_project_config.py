
import pytest

from core.project_config import ProjectConfig


def test_chat_only_project():
    config = ProjectConfig(
        name="chat_app",
        chat_enabled=True,
    )

    assert config.is_enabled("chat")
    assert not config.is_enabled("rag")


def test_rag_only_project():
    config = ProjectConfig(
        name="document_app",
        rag_enabled=True,
    )

    assert not config.is_enabled("chat")
    assert config.is_enabled("rag")


def test_chat_and_rag_project():
    config = ProjectConfig(
        name="general_assistant",
        chat_enabled=True,
        rag_enabled=True,
    )

    assert config.is_enabled("chat")
    assert config.is_enabled("rag")


def test_project_without_features_is_invalid():
    with pytest.raises(ValueError):
        ProjectConfig(name="empty_app")


def test_empty_project_name_is_invalid():
    with pytest.raises(ValueError):
        ProjectConfig(
            name=" ",
            chat_enabled=True,
        )


def test_unknown_feature_is_invalid():
    config = ProjectConfig(
        name="chat_app",
        chat_enabled=True,
    )

    with pytest.raises(ValueError):
        config.is_enabled("search")
