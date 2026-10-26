
import pickle

import faiss
import numpy as np
import pytest

from rag.vectorstore import VectorStore


def make_store(text="Original document"):
    store = VectorStore(dimension=2)

    store.add(
        vectors=[[1.0, 0.0]],
        documents=[
            {
                "text": text,
                "source": "sample.pdf",
                "chunk_id": 0,
            }
        ],
    )

    return store


def test_snapshot_round_trip(tmp_path):
    store = make_store()
    store.save(str(tmp_path))

    snapshot_path = tmp_path / "vectorstore.snapshot"

    assert snapshot_path.is_file()
    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "documents.pkl").is_file()

    loaded = VectorStore.load(str(tmp_path))

    assert loaded.index.ntotal == 1
    assert loaded.documents[0]["text"] == "Original document"
    assert loaded.search([1.0, 0.0])[0]["document"]["text"] == (
        "Original document"
    )


def test_failed_snapshot_replace_preserves_previous_snapshot(
    tmp_path,
    monkeypatch,
):
    store = make_store()
    store.save(str(tmp_path))

    snapshot_path = tmp_path / "vectorstore.snapshot"
    original_snapshot = snapshot_path.read_bytes()

    store.add(
        vectors=[[0.0, 1.0]],
        documents=[
            {
                "text": "New document",
                "source": "new.pdf",
                "chunk_id": 0,
            }
        ],
    )

    def fail_replace(source, destination):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(
        "rag.vectorstore.os.replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated disk write failure",
    ):
        store.save(str(tmp_path))

    # The previously committed snapshot must remain unchanged.
    assert snapshot_path.read_bytes() == original_snapshot

    loaded = VectorStore.load(str(tmp_path))

    assert loaded.index.ntotal == 1
    assert len(loaded.documents) == 1
    assert loaded.documents[0]["text"] == "Original document"


def test_load_supports_legacy_storage(tmp_path):
    store = make_store()

    faiss.write_index(
        store.index,
        str(tmp_path / "index.faiss"),
    )

    with open(
        tmp_path / "documents.pkl",
        "wb",
    ) as file:
        pickle.dump(store.documents, file)

    # No snapshot: the old storage format must still load.
    loaded = VectorStore.load(str(tmp_path))

    assert loaded.index.ntotal == 1
    assert loaded.documents[0]["text"] == "Original document"
