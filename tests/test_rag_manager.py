
from rag import rag_manager
from rag.vectorstore import VectorStore


def test_reindexing_same_pdf_does_not_duplicate_chunks(
    tmp_path,
    monkeypatch,
):
    pdf_path = str(tmp_path / "sample.pdf")
    storage_path = str(tmp_path / "storage")

    monkeypatch.setattr(
        rag_manager,
        "load_pdf",
        lambda path: "Sample PDF content",
    )

    monkeypatch.setattr(
        rag_manager,
        "split_text",
        lambda text, source: [
            {
                "text": "First chunk",
                "source": source,
                "chunk_id": 0,
            },
            {
                "text": "Second chunk",
                "source": source,
                "chunk_id": 1,
            },
        ],
    )

    monkeypatch.setattr(
        rag_manager,
        "create_embedding",
        lambda text: [1.0, 0.0],
    )

    # First indexing
    first_count = rag_manager.create_vector_store(
        pdf_path,
        storage_path=storage_path,
    )

    first_store = VectorStore.load(storage_path)
    assert first_count == 2
    assert first_store.index.ntotal == 2

    # Reindex the same PDF
    second_count = rag_manager.create_vector_store(
        pdf_path,
        storage_path=storage_path,
    )

    second_store = VectorStore.load(storage_path)

    assert second_count == 2
    assert second_store.index.ntotal == 2
    assert len(second_store.documents) == 2
    assert second_store.list_sources() == {
        pdf_path: 2,
    }
