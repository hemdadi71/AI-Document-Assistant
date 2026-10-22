
import os
import shutil
import uuid
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from core.project_config import ProjectConfig
from core.chat_service import ChatService
from core.assistant_service import AssistantService

from rag.rag_manager import create_vector_store
from rag.vectorstore import VectorStore
from rag.rag_service import RAGService

from llm import LLMServiceError
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Reusable AI Assistant Core")

STORAGE_PATH = "storage"
PDF_PATH = "data/pdf"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}")


PROJECT_CONFIG = ProjectConfig(
    name=os.getenv("PROJECT_NAME", "general_assistant"),
    chat_enabled=env_bool("CHAT_ENABLED", True),
    rag_enabled=env_bool("RAG_ENABLED", True),
)

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

vector_store: VectorStore | None = None
rag_service: RAGService | None = None

# In-memory conversation sessions for this application instance.
chat_sessions: dict[str, ChatService] = {}


def reload_rag():
    global vector_store, rag_service

    vector_store = VectorStore.load(STORAGE_PATH)
    rag_service = RAGService(vector_store)


if (
    PROJECT_CONFIG.rag_enabled
    and os.path.exists(os.path.join(STORAGE_PATH, "index.faiss"))
):
    reload_rag()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    mode: Literal["chat", "rag", "hybrid"] = "rag"

    conversation_id: str | None = Field(
        default=None,
        max_length=128,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


def get_chat_service(
    conversation_id: str | None,
) -> tuple[str, ChatService]:
    session_id = conversation_id or uuid.uuid4().hex

    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatService(
            system_prompt=SYSTEM_PROMPT,
        )

    return session_id, chat_sessions[session_id]


def build_assistant(
    chat_service: ChatService | None,
) -> AssistantService | None:
    # Chat can work even before the first PDF is uploaded.
    # AssistantService currently requires every enabled service
    # to be provided, so defer its creation while RAG has no index.
    if PROJECT_CONFIG.rag_enabled and rag_service is None:
        return None

    return AssistantService(
        config=PROJECT_CONFIG,
        chat_service=chat_service,
        rag_service=(
            rag_service if PROJECT_CONFIG.rag_enabled else None
        ),
    )


def execute_request(request: ChatRequest) -> dict:
    mode = request.mode

    if mode == "chat" and not PROJECT_CONFIG.chat_enabled:
        raise HTTPException(
            status_code=403,
            detail="Chat is disabled for this project.",
        )

    if mode == "rag" and not PROJECT_CONFIG.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if mode == "hybrid" and not (
        PROJECT_CONFIG.chat_enabled
        and PROJECT_CONFIG.rag_enabled
    ):
        raise HTTPException(
            status_code=403,
            detail="Hybrid mode requires Chat and RAG to be enabled.",
        )

    if mode in ("rag", "hybrid") and rag_service is None:
        raise HTTPException(
            status_code=409,
            detail="No documents uploaded yet.",
        )

    session_id = request.conversation_id
    chat_service = None

    if PROJECT_CONFIG.chat_enabled:
        session_id, chat_service = get_chat_service(session_id)

    assistant = build_assistant(chat_service)

    # Before a PDF is uploaded, ordinary chat still works.
    if assistant is None:
        if mode != "chat" or chat_service is None:
            raise HTTPException(
                status_code=409,
                detail="The required service is not ready.",
            )

        result = {
            "answer": chat_service.chat(request.question),
            "sources": [],
        }
    else:
        result = assistant.ask(
            request.question,
            mode=mode,
        )

    result["conversation_id"] = (
        session_id or request.conversation_id
    )

    return result


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = exc.errors()

    if any(
        error.get("loc") and error["loc"][-1] == "question"
        for error in errors
    ):
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "Question must not be empty and must be valid."
                )
            },
        )

    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request data."},
    )


def to_http_exception(exc: LLMServiceError) -> HTTPException:
    status_codes = {
        "configuration_error": 500,
        "authentication_error": 502,
        "rate_limit": 503,
        "timeout": 504,
        "connection_error": 502,
        "provider_error": 502,
        "empty_response": 502,
        "llm_error": 502,
    }

    return HTTPException(
        status_code=status_codes.get(exc.code, 502),
        detail={
            "code": exc.code,
            "message": str(exc),
        },
    )


def safe_stream(generator):
    try:
        yield from generator

    except LLMServiceError as exc:
        yield f"\n[LLM_ERROR:{exc.code}] {str(exc)}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": PROJECT_CONFIG.name,
        "features": {
            "chat": PROJECT_CONFIG.chat_enabled,
            "rag": PROJECT_CONFIG.rag_enabled,
            "hybrid": (
                PROJECT_CONFIG.chat_enabled
                and PROJECT_CONFIG.rag_enabled
            ),
        },
        "documents_ready": rag_service is not None,
    }


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not PROJECT_CONFIG.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    filename = os.path.basename(file.filename)

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    os.makedirs(PDF_PATH, exist_ok=True)

    file_path = os.path.join(PDF_PATH, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_count = create_vector_store(file_path)
        reload_rag()

    except Exception:
        # Avoid leaving a partially uploaded file behind.
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    return {
        "project": PROJECT_CONFIG.name,
        "filename": filename,
        "chunks": chunks_count,
        "status": "ready",
    }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return execute_request(request)

    except LLMServiceError as exc:
        raise to_http_exception(exc) from exc


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):
    # Preserve the existing RAG streaming behavior.
    if request.mode != "rag":
        raise HTTPException(
            status_code=400,
            detail="Streaming currently supports RAG mode only.",
        )

    if not PROJECT_CONFIG.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if rag_service is None:
        raise HTTPException(
            status_code=409,
            detail="No documents uploaded yet.",
        )

    try:
        generator = rag_service.stream_answer(request.question)

    except LLMServiceError as exc:
        raise to_http_exception(exc) from exc

    return StreamingResponse(
        safe_stream(generator),
        media_type="text/plain",
    )
