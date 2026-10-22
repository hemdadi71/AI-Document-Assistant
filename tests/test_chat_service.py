
import pytest

import core.chat_service as chat_module
from core.chat_service import ChatService


def test_chat_returns_answer(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        lambda messages: "Hello!",
    )

    service = ChatService()

    assert service.chat("Hi") == "Hello!"


def test_chat_preserves_conversation_history(monkeypatch):
    captured_messages = []

    def fake_response(messages):
        captured_messages.append(messages.copy())
        return f"Answer {len(captured_messages)}"

    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        fake_response,
    )

    service = ChatService()

    service.chat("First question")
    service.chat("Second question")

    second_request = captured_messages[1]

    assert len(second_request) == 3
    assert second_request[0]["content"] == "First question"
    assert second_request[1]["content"] == "Answer 1"
    assert second_request[2]["content"] == "Second question"


def test_projects_have_independent_histories(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        lambda messages: "Answer",
    )

    project_a = ChatService()
    project_b = ChatService()

    project_a.chat("Question A")

    assert len(project_a.history) == 2
    assert project_b.history == []


def test_system_prompt_is_sent_to_llm(monkeypatch):
    captured = {}

    def fake_response(messages):
        captured["messages"] = messages
        return "Answer"

    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        fake_response,
    )

    service = ChatService(
        system_prompt="You are a Python tutor."
    )

    service.chat("Explain lists")

    assert captured["messages"][0] == {
        "role": "system",
        "content": "You are a Python tutor.",
    }


def test_empty_question_is_rejected():
    service = ChatService()

    with pytest.raises(ValueError):
        service.chat("   ")


def test_failed_request_does_not_modify_history(monkeypatch):
    def fake_response(messages):
        raise RuntimeError("LLM failed")

    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        fake_response,
    )

    service = ChatService()

    with pytest.raises(RuntimeError):
        service.chat("Hello")

    assert service.history == []


def test_clear_history(monkeypatch):
    monkeypatch.setattr(
        chat_module,
        "get_ai_response",
        lambda messages: "Answer",
    )

    service = ChatService()
    service.chat("Hello")
    service.clear_history()

    assert service.history == []
