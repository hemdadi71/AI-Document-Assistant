
import json

from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.reranker import Reranker


QUESTIONS_PATH = "evaluation/questions.json"
STORAGE_PATH = "storage"

TARGET_QUESTIONS = {
    "What did the little prince ask the narrator to draw?",
    "Why was the narrator worried about the little prince's flower at the end of the story?",
}

with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)

store = VectorStore.load(STORAGE_PATH)
reranker = Reranker()

k = min(100, store.index.ntotal)

for item in questions:

    question = item["question"]

    if question not in TARGET_QUESTIONS:
        continue

    expected_ids = set(item["expected_chunk_ids"])
    query_vector = create_embedding(question)

    candidates = store.search(query_vector, k=k)

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    print("\nExpected chunks:", sorted(expected_ids))

    print("\n--- FAISS RANKS ---")

    found_ids = set()

    for rank, context in enumerate(candidates, start=1):

        document = context["document"]
        chunk_id = document["chunk_id"]

        if chunk_id in expected_ids:
            found_ids.add(chunk_id)
            print(
                f"Expected chunk {chunk_id}: "
                f"rank={rank}, "
                f"score={context['score']:.4f}"
            )

    missing = expected_ids - found_ids

    if missing:
        print("Expected chunks not found in FAISS results:", sorted(missing))

    reranked = reranker.rerank(
        question,
        candidates,
        top_k=len(candidates),
    )

    print("\n--- TOP 10 RERANKED ---")

    for rank, context in enumerate(reranked[:10], start=1):

        document = context["document"]

        print(
            f"Rank {rank}: "
            f"chunk={document['chunk_id']}, "
            f"faiss={context['score']:.4f}, "
            f"rerank={context['rerank_score']:.4f}"
        )

    print("\n--- EXPECTED CHUNKS AFTER RERANKING ---")

    for rank, context in enumerate(reranked, start=1):

        chunk_id = context["document"]["chunk_id"]

        if chunk_id in expected_ids:
            print(
                f"Chunk {chunk_id}: "
                f"rank={rank}, "
                f"rerank={context['rerank_score']:.4f}"
            )