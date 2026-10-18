
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag.rag_manager import create_vector_store
from rag.vectorstore import VectorStore
from rag.rag_service import RAGService
from llm import LLMServiceError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

STORAGE_PATH = "storage"
PDF_PATH = "data/pdf"


if os.path.exists(os.path.join(STORAGE_PATH, "index.faiss")):
    vector_store = VectorStore.load(STORAGE_PATH)
    rag_service = RAGService(vector_store)
else:
    vector_store = None
    rag_service = None


class ChatRequest(BaseModel):

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


def reload_rag():

    global vector_store, rag_service

    vector_store = VectorStore.load(STORAGE_PATH)
    rag_service = RAGService(vector_store)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = exc.errors()

    if any(error["loc"][-1] == "question" for error in errors):
        return JSONResponse(
            status_code=422,
            content={"detail": "Question must not be empty and must be valid."},
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


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

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

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_count = create_vector_store(file_path)

    reload_rag()

    return {
        "filename": filename,
        "chunks": chunks_count,
        "status": "ready",
    }


@app.post("/chat")
def chat(request: ChatRequest):

    if rag_service is None:
        raise HTTPException(
            status_code=409,
            detail="No documents uploaded yet.",
        )

    try:
        return rag_service.ask(request.question)

    except LLMServiceError as exc:
        raise to_http_exception(exc) from exc


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):

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
