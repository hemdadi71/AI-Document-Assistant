
import os

from rag.loader import load_pdf
from rag.splitter import split_text
from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore


def create_vector_store(
    pdf_path: str,
    storage_path: str = "storage",
) -> int:
    os.makedirs(storage_path, exist_ok=True)

    text = load_pdf(pdf_path)
    chunks = split_text(text, source=pdf_path)

    if not chunks:
        raise ValueError("No text chunks were extracted from PDF.")

    vectors = [
        create_embedding(chunk["text"])
        for chunk in chunks
    ]

    index_path = os.path.join(
        storage_path,
        "index.faiss",
    )

    if os.path.exists(index_path):
        store = VectorStore.load(storage_path)
    else:
        store = VectorStore(
            dimension=len(vectors[0]),
        )

    store.remove_source(pdf_path)

    store.add(vectors, chunks)
    store.save(storage_path)

    return len(chunks)
