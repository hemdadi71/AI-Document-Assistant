
import pytest

import core.project_runtime as runtime_module
from core.project_config import ProjectConfig
from core.project_runtime import ProjectRuntime


class FakeChatService:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt
        self.questions = []

    def chat(self, question):
        self.questions.append(question)
        return f"Answer: {question}"


def make_runtime(tmp_path, name):
    return ProjectRuntime(
        config=ProjectConfig(
            name=name,
            chat_enabled=True,
            rag_enabled=False,
        ),
        storage_root=str(tmp_path / "storage"),
        pdf_root=str(tmp_path / "pdfs"),
    )


def test_projects_have_separate_storage_paths(tmp_path):
    project_a = make_runtime(tmp_path, "project_a")
    project_b = make_runtime(tmp_path, "project_b")

    assert project_a.storage_path != project_b.storage_path
    assert project_a.pdf_path != project_b.pdf_path


def test_projects_have_independent_chat_histories(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runtime_module,
        "ChatService",
        FakeChatService,
    )

    project_a = make_runtime(tmp_path, "project_a")
    project_b = make_runtime(tmp_path, "project_b")

    result_a = project_a.ask(
        "Hello",
        mode="chat",
        conversation_id="session-1",
    )

    result_b = project_b.ask(
        "Hello",
        mode="chat",
        conversation_id="session-1",
    )

    assert result_a["conversation_id"] == "session-1"
    assert result_b["conversation_id"] == "session-1"

    assert (
        project_a.chat_sessions["session-1"]
        is not project_b.chat_sessions["session-1"]
    )


def test_chat_history_is_reused_within_project(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runtime_module,
        "ChatService",
        FakeChatService,
    )

    project = make_runtime(tmp_path, "project_a")

    project.ask(
        "First question",
        mode="chat",
        conversation_id="session-1",
    )

    project.ask(
        "Second question",
        mode="chat",
        conversation_id="session-1",
    )

    service = project.chat_sessions["session-1"]

    assert service.questions == [
        "First question",
        "Second question",
    ]


def test_disabled_chat_is_rejected(tmp_path):
    project = ProjectRuntime(
        config=ProjectConfig(
            name="documents",
            chat_enabled=False,
            rag_enabled=True,
        ),
        storage_root=str(tmp_path / "storage"),
        pdf_root=str(tmp_path / "pdfs"),
    )

    with pytest.raises(ValueError, match="Chat is disabled"):
        project.ask("Hello", mode="chat")


def test_rag_without_index_is_not_ready(tmp_path):
    project = ProjectRuntime(
        config=ProjectConfig(
            name="documents",
            chat_enabled=False,
            rag_enabled=True,
        ),
        storage_root=str(tmp_path / "storage"),
        pdf_root=str(tmp_path / "pdfs"),
    )

    with pytest.raises(RuntimeError, match="No documents"):
        project.ask("Question", mode="rag")
