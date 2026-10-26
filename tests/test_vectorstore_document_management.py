
from rag.vectorstore import VectorStore


def test_remove_source_removes_all_its_chunks(tmp_path):
    source_a = str(tmp_path / "a.pdf")
    source_b = str(tmp_path / "b.pdf")

    store = VectorStore(dimension=2)

    store.add(
        vectors=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        documents=[
            {"text": "A chunk 1", "source": source_a, "chunk_id": 0},
            {"text": "A chunk 2", "source": source_a, "chunk_id": 1},
            {"text": "B chunk 1", "source": source_b, "chunk_id": 0},
        ],
    )

    removed = store.remove_source(source_a)

    assert removed == 2
    assert store.index.ntotal == 1
    assert len(store.documents) == 1
    assert store.documents[0]["source"] == source_b

    store.save(str(tmp_path / "storage"))
    loaded = VectorStore.load(str(tmp_path / "storage"))

    assert loaded.index.ntotal == 1
    assert len(loaded.documents) == 1
    assert loaded.documents[0]["source"] == source_b


def test_remove_unknown_source_changes_nothing(tmp_path):
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0]],
        documents=[
            {"text": "Keep me", "source": "a.pdf", "chunk_id": 0}
        ],
    )

    removed = store.remove_source(
        str(tmp_path / "unknown.pdf")
    )

    assert removed == 0
    assert store.index.ntotal == 1
    assert len(store.documents) == 1
