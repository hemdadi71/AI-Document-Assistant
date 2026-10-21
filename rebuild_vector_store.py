
import glob
import os
import shutil
from datetime import datetime

from rag.loader import load_pdf
from rag.splitter import split_text
from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore


PDF_DIR = "data/pdf"
STORAGE_DIR = "storage"
TEMP_DIR = "storage_rebuilt_tmp"


def rebuild_vector_store():
    pdf_files = sorted(
        glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    all_documents = []
    all_vectors = []

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        text = load_pdf(pdf_path)

        chunks = split_text(
            text,
            source=pdf_path,
        )

        vectors = [
            create_embedding(chunk["text"])
            for chunk in chunks
        ]

        all_documents.extend(chunks)
        all_vectors.extend(vectors)

        print(
            f"Processed: {os.path.basename(pdf_path)} "
            f"| Chunks: {len(chunks)}"
        )

    if not all_vectors:
        raise ValueError("No text chunks were generated.")

    store = VectorStore(
        dimension=len(all_vectors[0])
    )

    store.add(all_vectors, all_documents)

    # Build the new index separately.
    store.save(TEMP_DIR)

    # Back up the current index before replacing it.
    backup_dir = None

    if os.path.exists(STORAGE_DIR):
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_dir = f"storage_backup_{timestamp}"

        shutil.copytree(
            STORAGE_DIR,
            backup_dir,
        )

        print(f"Backup created: {backup_dir}")

    old_dir = "storage_swap_old"

    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)

    try:
        if os.path.exists(STORAGE_DIR):
            os.replace(STORAGE_DIR, old_dir)

        os.replace(TEMP_DIR, STORAGE_DIR)

    except Exception:
        # Restore the old index if replacement fails.
        if (
            os.path.exists(old_dir)
            and not os.path.exists(STORAGE_DIR)
        ):
            os.replace(old_dir, STORAGE_DIR)

        raise

    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)

    print(
        f"\nVector Store rebuilt successfully. "
        f"Total chunks: {len(all_documents)}"
    )


if __name__ == "__main__":
    rebuild_vector_store()
