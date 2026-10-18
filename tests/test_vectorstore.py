
import numpy as np
import pytest

from rag.vectorstore import VectorStore


@pytest.fixture
def store():
    vector_store = VectorStore(dimension=2)

    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    documents = [
        {
            "text": "Document A",
            "source": "a.pdf",
            "chunk_id": 0,
        },
        {
            "text": "Document B",
            "source": "b.pdf",
            "chunk_id": 0,
        },
    ]

    vector_store.add(vectors, documents)

    return vector_store


def test_search_returns_all_results(store):
    results = store.search([1.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0]["document"]["text"] == "Document A"
    assert results[0]["score"] == pytest.approx(1.0)


def test_empty_store_returns_empty_list():
    store = VectorStore(dimension=2)

    assert store.search([1.0, 0.0]) == []


def test_search_handles_k_larger_than_index(store):
    results = store.search([1.0, 0.0], k=10)

    assert len(results) == 2


def test_search_with_invalid_k_returns_empty_list(store):
    assert store.search([1.0, 0.0], k=0) == []
    assert store.search([1.0, 0.0], k=-1) == []


def test_add_rejects_wrong_vector_dimension():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0, 0.5]],
            [{"text": "Document A"}],
        )


def test_add_rejects_mismatched_counts():
    store = VectorStore(dimension=2)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0], [0.0, 1.0]],
            [{"text": "Document A"}],
        )


def test_search_rejects_wrong_query_dimension(store):
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.5])


def test_save_and_load(store, tmp_path):
    store.save(str(tmp_path))

    loaded_store = VectorStore.load(str(tmp_path))

    results = loaded_store.search([1.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0]["document"]["text"] == "Document A"


def test_search_skips_invalid_faiss_indices(store):
    class FakeIndex:
        ntotal = 2

        def search(self, query, k):
            return (
                np.array([[0.9, 0.8]], dtype=np.float32),
                np.array([[0, -1]], dtype=np.int64),
            )

    store.index = FakeIndex()

    results = store.search([1.0, 0.0], k=2)

    assert len(results) == 1
    assert results[0]["document"]["text"] == "Document A"
