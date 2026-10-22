
from unittest.mock import Mock

import pytest

from core.assistant_service import AssistantService
from core.project_config import ProjectConfig


def test_combined_project_can_use_chat():
    chat = Mock()
    chat.chat.return_value = "Hello!"

    rag = Mock()

    config = ProjectConfig(
        name="general_assistant",
        chat_enabled=True,
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=chat,
        rag_service=rag,
    )

    result = assistant.ask("Hi", mode="chat")

    assert result == {
        "answer": "Hello!",
        "sources": [],
    }

    chat.chat.assert_called_once_with("Hi")
    rag.ask.assert_not_called()


def test_combined_project_can_use_rag():
    chat = Mock()

    rag = Mock()
    rag.ask.return_value = {
        "answer": "Forty-three sunsets.",
        "sources": [
            {
                "source": "book.pdf",
                "chunk_id": 42,
            }
        ],
    }

    config = ProjectConfig(
        name="general_assistant",
        chat_enabled=True,
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=chat,
        rag_service=rag,
    )

    result = assistant.ask(
        "How many sunsets?",
        mode="rag",
    )

    assert result["answer"] == "Forty-three sunsets."
    assert len(result["sources"]) == 1

    rag.ask.assert_called_once_with("How many sunsets?")
    chat.chat.assert_not_called()


def test_rag_only_project_rejects_chat():
    rag = Mock()

    config = ProjectConfig(
        name="document_assistant",
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        rag_service=rag,
    )

    with pytest.raises(ValueError, match="disabled"):
        assistant.ask("Hi", mode="chat")

    rag.ask.assert_not_called()


def test_chat_only_project_rejects_rag():
    chat = Mock()

    config = ProjectConfig(
        name="chat_assistant",
        chat_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=chat,
    )

    with pytest.raises(ValueError, match="disabled"):
        assistant.ask("Explain Python", mode="rag")

    chat.chat.assert_not_called()


def test_enabled_chat_requires_chat_service():
    config = ProjectConfig(
        name="chat_assistant",
        chat_enabled=True,
    )

    with pytest.raises(ValueError, match="Chat is enabled"):
        AssistantService(config=config)


def test_enabled_rag_requires_rag_service():
    config = ProjectConfig(
        name="document_assistant",
        rag_enabled=True,
    )

    with pytest.raises(ValueError, match="RAG is enabled"):
        AssistantService(config=config)


def test_empty_question_is_rejected():
    config = ProjectConfig(
        name="chat_assistant",
        chat_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=Mock(),
    )

    with pytest.raises(ValueError, match="empty"):
        assistant.ask("   ", mode="chat")


def test_hybrid_combines_chat_and_rag():
    chat = Mock()
    chat.chat.return_value = (
        "Document evidence: ...\nGeneral knowledge: ..."
    )

    rag = Mock()
    rag.retrieve_contexts.return_value = [
        {
            "document": {
                "text": "The little prince watched forty-three sunsets.",
                "source": "/documents/book.pdf",
                "chunk_id": 42,
            },
            "score": 0.8,
            "dense_score": 0.7,
            "bm25_score": 10.0,
            "hybrid_score": 0.03,
            "rerank_score": 8.0,
        }
    ]

    config = ProjectConfig(
        name="general_assistant",
        chat_enabled=True,
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=chat,
        rag_service=rag,
    )

    result = assistant.ask(
        "Explain the sunsets.",
        mode="hybrid",
    )

    assert "Document evidence" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "book.pdf"
    assert result["sources"][0]["chunk_id"] == 42

    rag.retrieve_contexts.assert_called_once_with(
        "Explain the sunsets."
    )

    chat.chat.assert_called_once()

    _, kwargs = chat.chat.call_args

    assert "forty-three sunsets" in kwargs["additional_context"]
    assert "General knowledge" in kwargs["additional_instructions"]


def test_hybrid_works_without_retrieved_contexts():
    chat = Mock()
    chat.chat.return_value = (
        "Document evidence: No relevant excerpts found.\n"
        "General knowledge: Hello!"
    )

    rag = Mock()
    rag.retrieve_contexts.return_value = []

    config = ProjectConfig(
        name="general_assistant",
        chat_enabled=True,
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=chat,
        rag_service=rag,
    )

    result = assistant.ask("Hi", mode="hybrid")

    assert result["sources"] == []

    _, kwargs = chat.chat.call_args

    assert (
        "No relevant document excerpts were retrieved."
        in kwargs["additional_context"]
    )


def test_hybrid_requires_both_features():
    rag = Mock()

    config = ProjectConfig(
        name="document_assistant",
        rag_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        rag_service=rag,
    )

    with pytest.raises(ValueError, match="requires both"):
        assistant.ask(
            "Explain this document.",
            mode="hybrid",
        )

    rag.retrieve_contexts.assert_not_called()


def test_unknown_mode_is_rejected():
    config = ProjectConfig(
        name="chat_assistant",
        chat_enabled=True,
    )

    assistant = AssistantService(
        config=config,
        chat_service=Mock(),
    )

    with pytest.raises(ValueError, match="Unsupported mode"):
        assistant.ask("Hello", mode="unknown")
