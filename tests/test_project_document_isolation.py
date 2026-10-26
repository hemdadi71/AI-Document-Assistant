
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main as main_module
import core.project_runtime as runtime_module

from core.project_config import ProjectConfig
from core.project_registry import ProjectRegistry
from core.project_runtime import ProjectRuntime
from core.security import api_key_env_name
from rag.vectorstore import VectorStore


PROJECT_A_KEY = "a" * 40
PROJECT_B_KEY = "b" * 40


def fake_create_vector_store(
    pdf_path: str,
    storage_path: str = "storage",
) -> int:
    """
    Replace PDF extraction and embedding generation with a
    deterministic vector while retaining real FAISS persistence.
    """
    storage = Path(storage_path)

    if (storage / "index.faiss").is_file():
        store = VectorStore.load(str(storage))
    else:
        store = VectorStore(dimension=2)

    store.remove_source(pdf_path)

    text = Path(pdf_path).read_bytes().decode(
        "utf-8",
        errors="replace",
    )

    store.add(
        vectors=[[1.0, 0.0]],
        documents=[
            {
                "text": text,
                "source": pdf_path,
                "chunk_id": 0,
            }
        ],
    )

    store.save(str(storage))

    return 1


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    project_a = ProjectConfig(
        name="project_a",
        chat_enabled=False,
        rag_enabled=True,
    )

    project_b = ProjectConfig(
        name="project_b",
        chat_enabled=False,
        rag_enabled=True,
    )

    monkeypatch.setenv(
        api_key_env_name("project_a"),
        PROJECT_A_KEY,
    )
    monkeypatch.setenv(
        api_key_env_name("project_b"),
        PROJECT_B_KEY,
    )

    storage_root = tmp_path / "storage"
    pdf_root = tmp_path / "data" / "projects"

    runtime_a = ProjectRuntime(
        config=project_a,
        storage_root=str(storage_root),
        pdf_root=str(pdf_root),
    )

    runtime_b = ProjectRuntime(
        config=project_b,
        storage_root=str(storage_root),
        pdf_root=str(pdf_root),
    )

    monkeypatch.setattr(
        runtime_module,
        "create_vector_store",
        fake_create_vector_store,
    )

    monkeypatch.setattr(
        main_module,
        "project_registry",
        ProjectRegistry(
            projects=[project_a, project_b],
        ),
    )

    monkeypatch.setattr(
        main_module,
        "project_runtimes",
        {
            "project_a": runtime_a,
            "project_b": runtime_b,
        },
    )

    client = TestClient(main_module.app)

    return {
        "client": client,
        "runtime_a": runtime_a,
        "runtime_b": runtime_b,
        "storage_root": storage_root,
        "pdf_root": pdf_root,
        "project_a": project_a,
        "project_b": project_b,
    }


def upload_pdf(client, api_key, filename, content):
    return client.post(
        "/upload",
        headers={"X-API-Key": api_key},
        files={
            "file": (
                filename,
                content,
                "application/pdf",
            )
        },
    )


def test_same_filename_is_isolated_between_projects(
    isolated_projects,
):
    client = isolated_projects["client"]
    runtime_a = isolated_projects["runtime_a"]
    runtime_b = isolated_projects["runtime_b"]

    # Upload a file with the same name to both projects.
    response_a = upload_pdf(
        client,
        PROJECT_A_KEY,
        "shared.pdf",
        b"Project A document",
    )

    response_b = upload_pdf(
        client,
        PROJECT_B_KEY,
        "shared.pdf",
        b"Project B document",
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    # Each project has its own physical PDF.
    pdf_a = Path(runtime_a.pdf_path) / "shared.pdf"
    pdf_b = Path(runtime_b.pdf_path) / "shared.pdf"

    assert pdf_a.is_file()
    assert pdf_b.is_file()
    assert pdf_a.read_bytes() != pdf_b.read_bytes()

    # Each project has its own storage directory and index.
    assert runtime_a.storage_path != runtime_b.storage_path

    sources_a = runtime_a.vector_store.list_sources()
    sources_b = runtime_b.vector_store.list_sources()

    assert sources_a == {str(pdf_a): 1}
    assert sources_b == {str(pdf_b): 1}

    # The API lists each project's own document.
    documents_a = client.get(
        "/documents",
        headers={"X-API-Key": PROJECT_A_KEY},
    )

    documents_b = client.get(
        "/documents",
        headers={"X-API-Key": PROJECT_B_KEY},
    )

    assert documents_a.status_code == 200
    assert documents_b.status_code == 200
    assert documents_a.json()["count"] == 1
    assert documents_b.json()["count"] == 1

    # Delete the document from project A only.
    delete_response = client.delete(
        "/documents/shared.pdf",
        headers={"X-API-Key": PROJECT_A_KEY},
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_deleted"] == 1

    # A's document is gone; B's document remains.
    assert not pdf_a.exists()
    assert pdf_b.is_file()

    remaining_a = client.get(
        "/documents",
        headers={"X-API-Key": PROJECT_A_KEY},
    )

    remaining_b = client.get(
        "/documents",
        headers={"X-API-Key": PROJECT_B_KEY},
    )

    assert remaining_a.status_code == 200
    assert remaining_b.status_code == 200
    assert remaining_a.json()["count"] == 0
    assert remaining_b.json()["count"] == 1

    assert runtime_a.vector_store.list_sources() == {}
    assert runtime_b.vector_store.list_sources() == {
        str(pdf_b): 1
    }


def test_project_indexes_remain_separate_after_runtime_reload(
    isolated_projects,
):
    client = isolated_projects["client"]

    upload_a = upload_pdf(
        client,
        PROJECT_A_KEY,
        "shared.pdf",
        b"Project A content",
    )

    upload_b = upload_pdf(
        client,
        PROJECT_B_KEY,
        "shared.pdf",
        b"Project B content",
    )

    assert upload_a.status_code == 200
    assert upload_b.status_code == 200

    # Simulate application Runtime reconstruction using the same
    # persistent storage and PDF directories.
    reloaded_a = ProjectRuntime(
        config=isolated_projects["project_a"],
        storage_root=str(isolated_projects["storage_root"]),
        pdf_root=str(isolated_projects["pdf_root"]),
    )

    reloaded_b = ProjectRuntime(
        config=isolated_projects["project_b"],
        storage_root=str(isolated_projects["storage_root"]),
        pdf_root=str(isolated_projects["pdf_root"]),
    )

    assert reloaded_a.vector_store is not None
    assert reloaded_b.vector_store is not None

    sources_a = reloaded_a.vector_store.list_sources()
    sources_b = reloaded_b.vector_store.list_sources()

    assert len(sources_a) == 1
    assert len(sources_b) == 1

    source_a = next(iter(sources_a))
    source_b = next(iter(sources_b))

    assert source_a != source_b
    assert Path(source_a).parent == Path(reloaded_a.pdf_path)
    assert Path(source_b).parent == Path(reloaded_b.pdf_path)

    assert reloaded_a.list_documents()[0]["filename"] == "shared.pdf"
    assert reloaded_b.list_documents()[0]["filename"] == "shared.pdf"

    # Both persisted stores contain their own document text.
    assert (
        reloaded_a.vector_store.documents[0]["text"]
        == "Project A content"
    )
    assert (
        reloaded_b.vector_store.documents[0]["text"]
        == "Project B content"
    )
