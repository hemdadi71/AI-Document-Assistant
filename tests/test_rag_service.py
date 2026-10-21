
import pytest
from unittest.mock import Mock

import rag.rag_service as rag_module
from rag.rag_service import RAGService


class FakeReranker:

    def __init__(self):
        self.called = False
        self.last_args = None

    def rerank(self, question, contexts, top_k=3):
        self.called = True
        self.last_args = (question, contexts, top_k)

        return [
            {
                **context,
                "rerank_score": float(10 - i),
            }
            for i, context in enumerate(contexts[:top_k])
        ]


@pytest.fixture
def service(monkeypatch):

    monkeypatch.setattr(rag_module, "Reranker", FakeReranker)
    monkeypatch.setattr(
        rag_module,
        "create_embedding",
        lambda question: [1.0, 0.0],
    )

    vector_store = Mock()
    vector_store.search.return_value = []

    return RAGService(vector_store)


def make_context(
    text="The answer is Fezzeh.",
    source="2.pdf",
    chunk_id=0,
    score=0.8,
):
    return {
        "document": {
            "text": text,
            "source": source,
            "chunk_id": chunk_id,
        },
        "score": score,
    }


def test_retrieve_returns_empty_when_no_candidates(service):

    contexts = service._retrieve_contexts("Who is the mother?")

    assert contexts == []
    assert service.reranker.called is False


def test_retrieve_skips_low_score_candidates(service, monkeypatch):

    monkeypatch.setattr(rag_module, "THRESHOLD", 0.1)

    service.vector_store.search.return_value = [
        make_context(score=0.05)
    ]

    contexts = service._retrieve_contexts("Who is the mother?")

    assert contexts == []
    assert service.reranker.called is False


def test_retrieve_reranks_candidates(service):

    service.vector_store.search.return_value = [
        make_context(score=0.8),
        make_context(
            text="Another document",
            source="3.pdf",
            score=0.5,
        ),
    ]

    contexts = service._retrieve_contexts("Who is the mother?")

    assert len(contexts) == 2
    assert service.reranker.called is True
    assert service.reranker.last_args[2] == 3
    assert contexts[0]["rerank_score"] == 10.0


def test_ask_returns_dont_know_without_contexts(
    service, monkeypatch
):

    monkeypatch.setattr(
        rag_module,
        "get_ai_response",
        lambda messages: pytest.fail(
            "LLM should not be called without contexts"
        ),
    )

    result = service.ask("Unknown question")

    assert result == {
        "answer": "I don't know.",
        "sources": [],
    }


def test_ask_returns_answer_and_sources(service, monkeypatch):

    service.vector_store.search.return_value = [
        make_context(
            source="/documents/2.pdf",
            chunk_id=4,
            score=0.8,
        )
    ]

    monkeypatch.setattr(
        rag_module,
        "build_rag_prompt",
        lambda question, contexts: "Test prompt",
    )

    monkeypatch.setattr(
        rag_module,
        "get_ai_response",
        lambda messages: "The answer is Fezzeh.",
    )

    result = service.ask("Who is the mother?")

    assert result["answer"] == "The answer is Fezzeh."
    assert result["sources"] == [
        {
            "source": "2.pdf",
            "chunk_id": 4,
            "score": 0.8,
            "rerank_score": 10.0,
        }
    ]


def test_stream_returns_dont_know_without_contexts(service):

    result = list(service.stream_answer("Unknown question"))

    assert result == ["I don't know."]


def test_stream_returns_llm_chunks(service, monkeypatch):

    service.vector_store.search.return_value = [
        make_context()
    ]

    monkeypatch.setattr(
        rag_module,
        "build_rag_prompt",
        lambda question, contexts: "Test prompt",
    )

    monkeypatch.setattr(
        rag_module,
        "stream_ai_response",
        lambda messages: iter(["The answer ", "is Fezzeh."]),
    )

    result = list(service.stream_answer("Who is the mother?"))

    assert result == ["The answer ", "is Fezzeh."]
