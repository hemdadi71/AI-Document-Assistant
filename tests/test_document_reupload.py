
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
import rag.rag_manager as rag_manager

from core.project_config import ProjectConfig
from core.project_registry import ProjectRegistry
from core.project_runtime import ProjectRuntime
from core.security import api_key_env_name
from rag.vectorstore import VectorStore

API_KEY = "a" * 40


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    project = ProjectConfig(
        name="reupload_test",
        chat_enabled=False,
        rag_enabled=True,
    )

    monkeypatch.setenv(
        api_key_env_name("reupload_test"),
        API_KEY,
    )

    # Use the real file saved by the upload endpoint.
    monkeypatch.setattr(
        rag_manager,
        "load_pdf",
        lambda path: Path(path).read_text(
            encoding="utf-8"
        ),
    )

    # Produce deterministic chunks from the extracted text.
    monkeypatch.setattr(
        rag_manager,
        "split_text",
        lambda text, source: [
            {
                "text": word,
                "source": source,
                "chunk_id": index,
            }
            for index, word in enumerate(text.split())
        ],
    )

    # Keep real FAISS indexing and persistence.
    monkeypatch.setattr(
        rag_manager,
        "create_embedding",
        lambda text: [1.0, 0.0],
    )

    storage_root = tmp_path / "storage"
    pdf_root = tmp_path / "data" / "projects"

    runtime = ProjectRuntime(
        config=project,
        storage_root=str(storage_root),
        pdf_root=str(pdf_root),
    )

    monkeypatch.setattr(
        main_module,
        "project_registry",
        ProjectRegistry(projects=[project]),
    )

    monkeypatch.setattr(
        main_module,
        "project_runtimes",
        {"reupload_test": runtime},
    )

    return {
        "client": TestClient(main_module.app),
        "runtime": runtime,
        "project": project,
        "storage_root": storage_root,
        "pdf_root": pdf_root,
    }


def upload(client, content):
    return client.post(
        "/upload",
        headers={"X-API-Key": API_KEY},
        files={
            "file": (
                "sample.pdf",
                content.encode("utf-8"),
                "application/pdf",
            )
        },
    )


def test_reupload_same_pdf_replaces_old_chunks(
    upload_client,
):
    client = upload_client["client"]
    runtime = upload_client["runtime"]

    # First upload: two chunks.
    first_response = upload(
        client,
        "alpha beta",
    )

    assert first_response.status_code == 200
    assert first_response.json()["chunks"] == 2

    first_store = runtime.vector_store

    assert first_store.index.ntotal == 2
    assert len(first_store.documents) == 2

    # Upload the same filename with different content.
    second_response = upload(
        client,
        "gamma delta epsilon",
    )

    assert second_response.status_code == 200
    assert second_response.json()["chunks"] == 3

    store = runtime.vector_store

    # The old chunks must be replaced, not appended.
    assert store.index.ntotal == 3
    assert len(store.documents) == 3

    texts = [
        document["text"]
        for document in store.documents
    ]

    assert texts == ["gamma", "delta", "epsilon"]
    assert "alpha" not in texts
    assert "beta" not in texts

    # All chunks belong to the same source.
    sources = store.list_sources()

    assert len(sources) == 1
    assert next(iter(sources.values())) == 3

    # The API must report one document with three chunks.
    documents_response = client.get(
        "/documents",
        headers={"X-API-Key": API_KEY},
    )

    assert documents_response.status_code == 200

    result = documents_response.json()

    assert result["count"] == 1
    assert result["documents"][0]["filename"] == "sample.pdf"
    assert result["documents"][0]["chunks"] == 3

    # The PDF on disk must contain the latest upload.
    pdf_path = Path(runtime.pdf_path) / "sample.pdf"

    assert pdf_path.read_text(
        encoding="utf-8"
    ) == "gamma delta epsilon"


def test_failed_reupload_preserves_previous_pdf_and_index(
    upload_client,
    monkeypatch,
):
    client = upload_client["client"]
    runtime = upload_client["runtime"]

    # Step 1: Upload a valid document.
    first_response = upload(
        client,
        "alpha beta",
    )

    assert first_response.status_code == 200

    pdf_path = Path(runtime.pdf_path) / "sample.pdf"
    snapshot_path = (
        Path(runtime.storage_path) / "vectorstore.snapshot"
    )

    original_pdf = pdf_path.read_bytes()
    original_snapshot = snapshot_path.read_bytes()

    original_store = VectorStore.load(
        runtime.storage_path
    )

    assert original_store.index.ntotal == 2
    assert [
        doc["text"] for doc in original_store.documents
    ] == ["alpha", "beta"]

    # Step 2: Simulate an embedding failure for the new upload.
    def fail_embedding(text):
        raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(
        rag_manager,
        "create_embedding",
        fail_embedding,
    )

    # Step 3: Re-upload the same filename with new content.
    # The endpoint should propagate the processing error.
    with pytest.raises(
        RuntimeError,
        match="Simulated embedding failure",
    ):
        upload(client, "gamma delta")

    # Step 4: The original PDF must be restored.
    assert pdf_path.is_file()
    assert pdf_path.read_bytes() == original_pdf

    # Step 5: The committed snapshot must remain unchanged.
    assert snapshot_path.read_bytes() == original_snapshot

    persisted_store = VectorStore.load(
        runtime.storage_path
    )

    assert persisted_store.index.ntotal == 2
    assert len(persisted_store.documents) == 2
    assert [
        doc["text"] for doc in persisted_store.documents
    ] == ["alpha", "beta"]

    # Step 6: The in-memory Runtime must still expose the
    # previously valid index.
    assert runtime.vector_store.index.ntotal == 2
    assert [
        doc["text"] for doc in runtime.vector_store.documents
    ] == ["alpha", "beta"]


def test_failed_reupload_recovers_after_runtime_restart(
    upload_client,
    monkeypatch,
):
    client = upload_client["client"]
    runtime = upload_client["runtime"]

    # Upload the original document.
    response = upload(client, "alpha beta")

    assert response.status_code == 200

    pdf_path = Path(runtime.pdf_path) / "sample.pdf"
    original_pdf = pdf_path.read_bytes()

    # Simulate failure while processing a replacement document.
    def fail_embedding(text):
        raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(
        rag_manager,
        "create_embedding",
        fail_embedding,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated embedding failure",
    ):
        upload(client, "gamma delta")

    # Reconstruct the Runtime as if the application restarted.
    recovered_runtime = ProjectRuntime(
        config=upload_client["project"],
        storage_root=str(upload_client["storage_root"]),
        pdf_root=str(upload_client["pdf_root"]),
    )

    # The original PDF must still exist.
    assert pdf_path.is_file()
    assert pdf_path.read_bytes() == original_pdf

    # The persisted index must load successfully.
    store = recovered_runtime.vector_store

    assert store is not None
    assert store.index.ntotal == 2
    assert len(store.documents) == 2

    assert [
        document["text"]
        for document in store.documents
    ] == ["alpha", "beta"]

    assert next(iter(store.list_sources().values())) == 2
