
import os

from rag.vectorstore import VectorStore
from rag.reranker import Reranker


QUESTION = "What did the little prince ask the narrator to draw?"
SOURCE = "The-Little-Prince-for-Level-2-PDF.pdf"

store = VectorStore.load("storage")
reranker = Reranker()

target_docs = {}

for doc in store.documents:

    if os.path.basename(doc.get("source", "")) != SOURCE:
        continue

    chunk_id = doc.get("chunk_id")

    if chunk_id in {7, 8}:
        target_docs[chunk_id] = doc

if not {7, 8}.issubset(target_docs):
    raise RuntimeError("Expected chunks 7 and 8 were not found.")

text_7 = target_docs[7]["text"]
text_8 = target_docs[8]["text"]


tests = [
    ("Chunk 7", text_7),
    ("Chunk 8", text_8),
    ("Chunks 7 + 8", text_7 + "\n" + text_8),

    (
        "Relevant excerpt from source",
        """
        A funny little voice woke me up in the morning.
        It said, "Please, draw me a sheep!"
        "What?"
        "Draw me a sheep!"
        I jumped to my feet as if I was hit by lightning.
        """
    ),

    (
        "Direct answer",
        "The little prince asked the narrator: "
        "'Please, draw me a sheep!'"
    ),
]

pairs = [(QUESTION, text) for _, text in tests]
scores = reranker.model.predict(pairs)

print("\nRERANKER DIAGNOSTIC")
print("=" * 60)

for (name, text), score in zip(tests, scores):
    print(f"\n{name}")
    print(f"Score: {float(score):.4f}")


pairs = [
    (QUESTION, text)
    for _, text in tests
]

scores = reranker.model.predict(pairs)

print("\nRERANKER DIAGNOSTIC")
print("=" * 60)

for (name, text), score in zip(tests, scores):
    print(f"\n{name}")
    print(f"Score: {float(score):.4f}")
    print(f"Text preview: {text[:250]!r}")
