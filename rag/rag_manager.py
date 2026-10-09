import os

from rag.loader import load_pdf
from rag.splitter import split_text
from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore


def create_vector_store(pdf_path: str):

    text = load_pdf(pdf_path)

    chunks = split_text(text, source=pdf_path)

    vectors = [create_embedding(chunk["text"]) for chunk in chunks]

    if os.path.exists("storage/index.faiss"):

        store = VectorStore.load("storage")

    else:

        store = VectorStore(dimension=len(vectors[0]))

    store.add(vectors, chunks)

    store.save("storage")

    return len(chunks)
