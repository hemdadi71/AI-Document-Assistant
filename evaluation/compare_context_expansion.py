
import os

from sentence_transformers import CrossEncoder

from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.hybrid_retriever import HybridRetriever


SOURCE = "The-Little-Prince-for-Level-2-PDF.pdf"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K = 100
BATCH_SIZE = 16

TEST_CASES = [
    {
        "question": "What did the narrator see in a book when he was six years old?",
        "expected_chunk_ids": [0],
    },
    {
        "question": "How many sunsets did the little prince watch in one day?",
        "expected_chunk_ids": [42],
    },
    {
        "question": "What did the little prince ask the narrator to draw?",
        "expected_chunk_ids": [7, 8],
    },
    {
        "question": "What did the little prince's flower ask him to bring?",
        "expected_chunk_ids": [55],
    },
    {
        "question": "Why did the man on the second planet want people to admire him?",
        "expected_chunk_ids": [84],
    },
    {
        "question": "What did the king say he ruled over?",
        "expected_chunk_ids": [72],
    },
    {
        "question": "What would the little prince do with fifty-three extra minutes?",
        "expected_chunk_ids": [154],
    },
    {
        "question": "Why was the narrator worried about the little prince's flower at the end of the story?",
        "expected_chunk_ids": [192, 193],
    },
]


def document_key(document):
    return (
        os.path.basename(document.get("source", "")),
        document.get("chunk_id"),
    )


def build_document_lookup(documents):
    return {
        document_key(document): document
        for document in documents
    }


def expand_context(document, document_lookup):

    source, chunk_id = document_key(document)

    window_ids = []
    parts = []

    # Previous chunk, current chunk, next chunk
    for neighbor_id in (
        chunk_id - 1,
        chunk_id,
        chunk_id + 1,
    ):

        neighbor = document_lookup.get(
            (source, neighbor_id)
        )

        if neighbor is None:
            continue

        window_ids.append(neighbor_id)

        label = (
            "CURRENT CHUNK"
            if neighbor_id == chunk_id
            else f"CHUNK {neighbor_id}"
        )

        parts.append(
            f"[{label}]\n{neighbor['text']}"
        )

    expanded_text = "\n\n".join(parts)

    return expanded_text, window_ids


def calculate_metrics(ranked_results, expected_ids, k):

    covered_ids = set()

    for result in ranked_results[:k]:
        covered_ids.update(result["window_ids"])

    recall = (
        len(covered_ids & expected_ids)
        / len(expected_ids)
    )

    return recall


def calculate_mrr(ranked_results, expected_ids):

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):

        if expected_ids.intersection(
            result["window_ids"]
        ):
            return 1.0 / rank

    return 0.0


def calculate_exact_metrics(ranked_results, expected_ids):

    # In the baseline, each result contains one chunk.
    exact_results = [
        {
            **result,
            "window_ids": [
                result["document"]["chunk_id"]
            ],
        }
        for result in ranked_results
    ]

    return {
        "recall@3": calculate_metrics(
            exact_results, expected_ids, 3
        ),
        "recall@10": calculate_metrics(
            exact_results, expected_ids, 10
        ),
        "mrr": calculate_mrr(
            exact_results, expected_ids
        ),
    }


def evaluate_question(
    question,
    expected_chunk_ids,
    retriever,
    model,
    document_lookup,
):

    query_vector = create_embedding(question)

    candidates = retriever.search(
        question=question,
        query_vector=query_vector,
        k=RETRIEVAL_K,
    )

    expected_ids = set(expected_chunk_ids)

    candidate_ids = {
        document["chunk_id"]
        for item in candidates
        if (
            os.path.basename(
                item["document"].get("source", "")
            ) == SOURCE
        )
        for document in [item["document"]]
    }

    candidate_recall = (
        len(candidate_ids & expected_ids)
        / len(expected_ids)
    )

    original_pairs = [
        (question, item["document"]["text"])
        for item in candidates
    ]

    expanded_contexts = []

    for item in candidates:

        text, window_ids = expand_context(
            item["document"],
            document_lookup,
        )

        expanded_contexts.append({
            "document": item["document"],
            "window_ids": window_ids,
            "text": text,
        })

    expanded_pairs = [
        (question, item["text"])
        for item in expanded_contexts
    ]

    original_scores = model.predict(
        original_pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )

    expanded_scores = model.predict(
        expanded_pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
    )

    baseline = [
        {
            "document": item["document"],
            "window_ids": [
                item["document"]["chunk_id"]
            ],
            "score": float(score),
        }
        for item, score in zip(
            candidates, original_scores
        )
    ]

    expanded = [
        {
            "document": item["document"],
            "window_ids": item["window_ids"],
            "score": float(score),
        }
        for item, score in zip(
            expanded_contexts, expanded_scores
        )
    ]

    baseline.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    expanded.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    baseline_metrics = calculate_exact_metrics(
        baseline, expected_ids
    )

    expanded_metrics = {
        "recall@3": calculate_metrics(
            expanded, expected_ids, 3
        ),
        "recall@10": calculate_metrics(
            expanded, expected_ids, 10
        ),
        "mrr": calculate_mrr(
            expanded, expected_ids
        ),
    }

    print("\n" + "=" * 75)
    print(f"QUESTION: {question}")
    print(f"Expected chunks: {expected_chunk_ids}")
    print(f"Candidate Recall: {candidate_recall:.3f}")

    print("\nBASELINE: ORIGINAL CHUNKS")

    print(
        f"Recall@3: {baseline_metrics['recall@3']:.3f} | "
        f"Recall@10: {baseline_metrics['recall@10']:.3f} | "
        f"MRR: {baseline_metrics['mrr']:.3f}"
    )

    for rank, result in enumerate(baseline[:3], start=1):

        print(
            f"  {rank}. chunk="
            f"{result['document']['chunk_id']} | "
            f"score={result['score']:.4f}"
        )

    print("\nCONTEXT EXPANSION: PREVIOUS + CURRENT + NEXT")

    print(
        f"Window Recall@3: "
        f"{expanded_metrics['recall@3']:.3f} | "
        f"Window Recall@10: "
        f"{expanded_metrics['recall@10']:.3f} | "
        f"Window MRR: {expanded_metrics['mrr']:.3f}"
    )

    for rank, result in enumerate(expanded[:3], start=1):

        print(
            f"  {rank}. center_chunk="
            f"{result['document']['chunk_id']} | "
            f"window={result['window_ids']} | "
            f"score={result['score']:.4f}"
        )

    return baseline_metrics, expanded_metrics


def main():

    vector_store = VectorStore.load("storage")

    retriever = HybridRetriever(vector_store)

    document_lookup = build_document_lookup(
        vector_store.documents
    )

    print(f"Loading model: {MODEL_NAME}")

    model = CrossEncoder(MODEL_NAME)

    baseline_results = []
    expanded_results = []

    for case in TEST_CASES:

        baseline, expanded = evaluate_question(
            question=case["question"],
            expected_chunk_ids=case["expected_chunk_ids"],
            retriever=retriever,
            model=model,
            document_lookup=document_lookup,
        )

        baseline_results.append(baseline)
        expanded_results.append(expanded)

    print("\n" + "=" * 75)
    print("AGGREGATE COMPARISON")
    print("=" * 75)

    for metric in ("recall@3", "recall@10", "mrr"):

        baseline_avg = sum(
            result[metric]
            for result in baseline_results
        ) / len(baseline_results)

        expanded_avg = sum(
            result[metric]
            for result in expanded_results
        ) / len(expanded_results)

        print(f"\n{metric}")
        print(f"  Original chunks: {baseline_avg:.4f}")
        print(f"  Expanded context: {expanded_avg:.4f}")


if __name__ == "__main__":
    main()
