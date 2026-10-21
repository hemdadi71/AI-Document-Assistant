
from pathlib import Path

from rag.vectorstore import VectorStore


SOURCE = "The-Little-Prince-for-Level-2-PDF.pdf"

SEARCH_TERMS = [
    "draw me a sheep",
    "please, draw me a sheep",
    "glass over the flower",
    "sheep can come to the flower",
    "ate or didn't eat a rose",
]


def main():
    store = VectorStore.load("storage")

    book_chunks = [
        doc
        for doc in store.documents
        if Path(doc.get("source", "")).name == SOURCE
    ]

    print(f"Total chunks in index: {len(store.documents)}")
    print(f"Book chunks: {len(book_chunks)}")

    for term in SEARCH_TERMS:
        print("\n" + "=" * 70)
        print(f"SEARCH TERM: {term}")

        matches = []

        for doc in book_chunks:
            text = doc.get("text", "")
            position = text.lower().find(term.lower())

            if position != -1:
                matches.append((doc, position))

        if not matches:
            print("NOT FOUND as an exact phrase.")
            continue

        for doc, position in matches:
            text = doc["text"]

            start = max(0, position - 180)
            end = min(len(text), position + len(term) + 220)

            print(
                f"\nChunk ID: {doc['chunk_id']}"
                f"\nText length: {len(text)}"
                f"\nEvidence: {text[start:end]}"
            )


if __name__ == "__main__":
    main()
