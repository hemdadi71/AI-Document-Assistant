
import json
import os
from pathlib import Path

from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.rag_service import (
    RAGService,
    RETRIEVAL_K,
    FINAL_CONTEXTS_K,
    THRESHOLD,
)


QUESTIONS_PATH = Path("evaluation/questions.json")
RESULTS_PATH = Path("evaluation/retrieval_results.json")
STORAGE_PATH = "storage"


def get_source(document):
    return os.path.basename(document.get("source", ""))


def get_chunk_id(context):
    return context["document"].get("chunk_id")


def main():

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    vector_store = VectorStore.load(STORAGE_PATH)
    rag_service = RAGService(vector_store)

    results = []

    source_hits = 0
    chunk_retrieval_hits = 0
    chunk_reranking_hits = 0
    chunk_evaluable = 0

    for number, item in enumerate(questions, start=1):

        question = item["question"]
        expected_source = Path(item["expected_source"]).name
        expected_chunk_ids = set(item.get("expected_chunk_ids", []))

        query_vector = create_embedding(question)

        candidates = vector_store.search(
            query_vector,
            k=RETRIEVAL_K,
        )

        # Apply the same threshold rule as RAGService.
        if (
            not candidates
            or candidates[0]["score"] < THRESHOLD
        ):
            final_contexts = []
        else:
            final_contexts = rag_service.reranker.rerank(
                question,
                candidates,
                top_k=FINAL_CONTEXTS_K,
            )

        candidate_sources = [
            get_source(item["document"])
            for item in candidates
        ]

        final_sources = [
            get_source(item["document"])
            for item in final_contexts
        ]

        candidate_chunk_ids = [
            get_chunk_id(item)
            for item in candidates
        ]

        final_chunk_ids = [
            get_chunk_id(item)
            for item in final_contexts
        ]

        source_hit = expected_source in candidate_sources
        source_hits += int(source_hit)

        if expected_chunk_ids:
            chunk_evaluable += 1

            retrieval_hit = bool(
                expected_chunk_ids.intersection(
                    candidate_chunk_ids
                )
            )

            reranking_hit = bool(
                expected_chunk_ids.intersection(
                    final_chunk_ids
                )
            )

            chunk_retrieval_hits += int(retrieval_hit)
            chunk_reranking_hits += int(reranking_hit)
        else:
            retrieval_hit = None
            reranking_hit = None

        print(f"\n{'=' * 60}")
        print(f"Question {number}: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Source retrieved: {source_hit}")
        print(f"Expected chunks: {sorted(expected_chunk_ids)}")
        print(f"Chunk retrieved: {retrieval_hit}")
        print(f"Chunk survived reranking: {reranking_hit}")

        print("\nTop FAISS candidates:")
        for rank, context in enumerate(candidates[:5], start=1):
            print(
                f"  Rank {rank}: "
                f"chunk={get_chunk_id(context)}, "
                f"score={context['score']:.4f}"
            )

        print("\nFinal contexts:")
        for rank, context in enumerate(final_contexts, start=1):
            print(
                f"  Rank {rank}: "
                f"chunk={get_chunk_id(context)}, "
                f"rerank_score={context['rerank_score']:.4f}"
            )

        results.append({
            "question": question,
            "expected_answer": item["expected_answer"],
            "expected_source": expected_source,
            "expected_chunk_ids": sorted(expected_chunk_ids),
            "candidate_chunk_ids": candidate_chunk_ids,
            "final_chunk_ids": final_chunk_ids,
            "source_retrieved": source_hit,
            "expected_chunk_retrieved": retrieval_hit,
            "expected_chunk_after_reranking": reranking_hit,
        })

    total = len(questions)

    print(f"\n{'=' * 60}")
    print(f"Total questions: {total}")

    if total:
        print(
            "Expected source retrieval rate: "
            f"{source_hits / total:.1%}"
        )

    if chunk_evaluable:
        print(
            "Expected chunk retrieval rate: "
            f"{chunk_retrieval_hits / chunk_evaluable:.1%}"
        )
        print(
            "Expected chunk retention after reranking: "
            f"{chunk_reranking_hits / chunk_evaluable:.1%}"
        )
    else:
        print("No expected chunk IDs available for evaluation.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
