
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
