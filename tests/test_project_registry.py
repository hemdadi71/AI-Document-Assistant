
import pytest

from core.project_config import ProjectConfig
from core.project_registry import ProjectRegistry


def test_register_and_get_project():
    project = ProjectConfig(
        name="documents",
        rag_enabled=True,
    )

    registry = ProjectRegistry([project])

    assert registry.get("documents") == project
    assert registry.contains("documents")


def test_projects_have_independent_features():
    chat_project = ProjectConfig(
        name="chat",
        chat_enabled=True,
    )

    rag_project = ProjectConfig(
        name="rag",
        rag_enabled=True,
    )

    registry = ProjectRegistry([
        chat_project,
        rag_project,
    ])

    assert registry.get("chat").chat_enabled is True
    assert registry.get("chat").rag_enabled is False

    assert registry.get("rag").chat_enabled is False
    assert registry.get("rag").rag_enabled is True


def test_duplicate_project_is_rejected():
    project = ProjectConfig(
        name="documents",
        rag_enabled=True,
    )

    registry = ProjectRegistry([project])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(project)


def test_unknown_project_is_rejected():
    registry = ProjectRegistry()

    with pytest.raises(KeyError, match="Unknown project"):
        registry.get("unknown")


def test_list_projects():
    registry = ProjectRegistry([
        ProjectConfig(name="chat", chat_enabled=True),
        ProjectConfig(name="rag", rag_enabled=True),
    ])

    projects = registry.list_projects()

    assert len(projects) == 2
    assert {project.name for project in projects} == {
        "chat",
        "rag",
    }
