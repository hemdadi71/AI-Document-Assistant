
import os
import uuid
from typing import Literal

from core.project_config import ProjectConfig
from core.chat_service import ChatService
from core.assistant_service import AssistantService

from rag.rag_manager import create_vector_store
from rag.vectorstore import VectorStore
from rag.rag_service import RAGService


RequestMode = Literal["chat", "rag", "hybrid"]


class ProjectRuntime:

    def __init__(
        self,
        config: ProjectConfig,
        storage_root: str = "storage",
        pdf_root: str = "data/projects",
        system_prompt: str | None = None,
    ):
        self.config = config

        self.storage_path = os.path.join(
            storage_root,
            "projects",
            config.name,
        )

        self.pdf_path = os.path.join(
            pdf_root,
            config.name,
            "pdf",
        )

        self.system_prompt = system_prompt

        # Conversation history belongs to this project only.
        self.chat_sessions: dict[str, ChatService] = {}

        self.vector_store: VectorStore | None = None
        self.rag_service: RAGService | None = None

        self.reload_rag()

    def reload_rag(self) -> None:
        index_path = os.path.join(
            self.storage_path,
            "index.faiss",
        )

        if (
            not self.config.rag_enabled
            or not os.path.exists(index_path)
        ):
            self.vector_store = None
            self.rag_service = None
            return

        self.vector_store = VectorStore.load(
            self.storage_path,
        )

        self.rag_service = RAGService(
            self.vector_store,
        )

    def index_pdf(self, pdf_path: str) -> int:
        if not self.config.rag_enabled:
            raise ValueError(
                "RAG is disabled for this project."
            )

        chunks_count = create_vector_store(
            pdf_path,
            storage_path=self.storage_path,
        )

        self.reload_rag()

        return chunks_count

    
    def list_documents(self) -> list[dict]:
        """List indexed PDF documents for this project."""

        if not self.config.rag_enabled:
            raise ValueError(
                "RAG is disabled for this project."
            )

        if self.vector_store is None:
            return []

        sources = self.vector_store.list_sources()
        documents = {}

        for source, chunks_count in sources.items():
            normalized_source = os.path.normcase(
                os.path.abspath(source)
            )

            documents[normalized_source] = {
                "filename": os.path.basename(source),
                "chunks": chunks_count,
                "status": "ready",
            }

        return sorted(
            documents.values(),
            key=lambda document: document["filename"].lower(),
        )

    
    def delete_document(self, filename: str) -> int:
        """Delete a document's indexed chunks from this project."""

        if not self.config.rag_enabled:
            raise ValueError(
                "RAG is disabled for this project."
            )

        if (
            not filename
            or filename != os.path.basename(filename)
            or filename in {".", ".."}
            or not filename.lower().endswith(".pdf")
        ):
            raise ValueError("Invalid PDF filename.")

        if self.vector_store is None:
            raise FileNotFoundError(
                f"Document not found: {filename}"
            )

        sources = self.vector_store.list_sources()

        matching_sources = [
            source
            for source in sources
            if os.path.basename(
                os.path.normpath(source)
            ) == filename
        ]

        if not matching_sources:
            raise FileNotFoundError(
                f"Indexed document not found: {filename}"
            )

        if len(matching_sources) > 1:
            raise ValueError(
                "Multiple indexed sources have this filename. "
                "Delete by a unique document ID instead."
            )

        source_path = matching_sources[0]
        source_abs = os.path.abspath(source_path)
        project_pdf_dir = os.path.abspath(self.pdf_path)

        try:
            is_project_owned = (
                os.path.commonpath(
                    [source_abs, project_pdf_dir]
                )
                == project_pdf_dir
            )
        except ValueError:
            is_project_owned = False

        # Only delete the physical PDF if it belongs to this project.
        pdf_path = (
            source_abs
            if is_project_owned and os.path.isfile(source_abs)
            else None
        )

        backup_path = None

        if pdf_path:
            backup_path = (
                f"{pdf_path}.delete-backup-{uuid.uuid4().hex}"
            )
            os.replace(pdf_path, backup_path)

        try:
            # Use the exact source path recorded in the index.
            removed_count = self.vector_store.remove_source(
                source_path
            )

            if removed_count == 0:
                raise FileNotFoundError(
                    f"No indexed chunks found for: {filename}"
                )

            self.vector_store.save(self.storage_path)

        except Exception:
            self.reload_rag()

            if (
                backup_path
                and os.path.isfile(backup_path)
            ):
                os.replace(backup_path, pdf_path)

            raise

        if backup_path and os.path.isfile(backup_path):
            try:
                os.remove(backup_path)
            except OSError:
                pass

        self.reload_rag()

        return removed_count



    def get_chat_service(
        self,
        conversation_id: str | None = None,
    ) -> tuple[str, ChatService]:
        session_id = conversation_id or uuid.uuid4().hex

        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = ChatService(
                system_prompt=self.system_prompt,
            )

        return session_id, self.chat_sessions[session_id]

    def ask(
        self,
        question: str,
        *,
        mode: RequestMode,
        conversation_id: str | None = None,
    ) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if mode not in ("chat", "rag", "hybrid"):
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "chat":
            if not self.config.chat_enabled:
                raise ValueError(
                    "Chat is disabled for this project."
                )

            session_id, chat_service = self.get_chat_service(
                conversation_id,
            )

            result = {
                "answer": chat_service.chat(question),
                "sources": [],
            }

        elif mode == "rag":
            if not self.config.rag_enabled:
                raise ValueError(
                    "RAG is disabled for this project."
                )

            if self.rag_service is None:
                raise RuntimeError(
                    "No documents are indexed for this project."
                )

            result = self.rag_service.ask(question)
            session_id = None

        else:
            if not (
                self.config.chat_enabled
                and self.config.rag_enabled
            ):
                raise ValueError(
                    "Hybrid requires Chat and RAG to be enabled."
                )

            if self.rag_service is None:
                raise RuntimeError(
                    "No documents are indexed for this project."
                )

            session_id, chat_service = self.get_chat_service(
                conversation_id,
            )

            assistant = AssistantService(
                config=self.config,
                chat_service=chat_service,
                rag_service=self.rag_service,
            )

            result = assistant.ask(
                question,
                mode="hybrid",
            )

        if session_id is not None:
            result["conversation_id"] = session_id

        return result

    def stream_answer(self, question: str):
        if not self.config.rag_enabled:
            raise ValueError(
                "RAG is disabled for this project."
            )

        if self.rag_service is None:
            raise RuntimeError(
                "No documents are indexed for this project."
            )

        return self.rag_service.stream_answer(question)
