from fastapi import FastAPI, UploadFile, File
import shutil
import os
from rag.rag_manager import create_vector_store
from pydantic import BaseModel
from rag.vectorstore import VectorStore
from rag.rag_service import RAGService
from fastapi.responses import StreamingResponse

app = FastAPI()

if os.path.exists("storage/index.faiss"):

    vector_store = VectorStore.load("storage")
    rag_service = RAGService(vector_store)

else:

    vector_store = None
    rag_service = None


class ChatRequest(BaseModel):
    question: str


def reload_rag():

    global vector_store, rag_service

    vector_store = VectorStore.load("storage")

    rag_service = RAGService(vector_store)


# .................................................................................
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks_count = create_vector_store(file_path)

    reload_rag()

    return {"filename": file.filename, "chunks": chunks_count, "status": "ready"}


@app.post("/chat")
def chat(request: ChatRequest):

    result = rag_service.ask(request.question)

    return result


@app.post("/chat-stream")
def chat_stream(request: ChatRequest):

    generator = rag_service.stream_answer(request.question)

    return StreamingResponse(generator, media_type="text/plain")
