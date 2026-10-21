import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.hybrid_retriever import HybridRetriever
from evaluation.evidence_metrics import (
    document_key,
    first_complete_evidence_rank,
    hit_at,
    mrr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.json")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K = 100
RRF_K = 60
BATCH_SIZE = 16


def load_test_cases():
    """Load the canonical evaluation set; relevance is text-based, not chunk-ID-based."""
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    for index, case in enumerate(cases):
        if not case.get("question"):
            raise ValueError(f"Evaluation case {index} has no question.")
        if not case.get("expected_source"):
            raise ValueError(f"Evaluation case {index} has no expected_source.")
        if not case.get("expected_evidence"):
            raise ValueError(
                f"Evaluation case {index} has no expected_evidence."
            )

    return cases


def reciprocal_rank_fusion(hybrid, reranked):
    """Combine Hybrid Retrieval and Cross-Encoder positions with RRF."""
    hybrid_ranks = {
        document_key(item): rank
        for rank, item in enumerate(hybrid, start=1)
    }
    reranker_ranks = {
        document_key(item): rank
        for rank, item in enumerate(reranked, start=1)
    }

    fused = []

    for item in hybrid:
        key = document_key(item)
        hybrid_rank = hybrid_ranks[key]
        reranker_rank = reranker_ranks.get(key)

        fusion_score = 1.0 / (RRF_K + hybrid_rank)

        if reranker_rank is not None:
            fusion_score += 1.0 / (RRF_K + reranker_rank)

        fused.append({
            **item,
            "fusion_score": fusion_score,
        })

    fused.sort(
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    return fused


def evaluate_strategy(name, results, case):
    metrics = {
        "hit@3": hit_at(results, case, 3),
        "hit@10": hit_at(results, case, 10),
        "mrr": mrr(results, case),
    }

    print(
        f"{name:<16} "
        f"Hit@3={metrics['hit@3']:.3f} | "
        f"Hit@10={metrics['hit@10']:.3f} | "
        f"MRR={metrics['mrr']:.4f}"
    )

    rank = first_complete_evidence_rank(results, case)
    print(f"  First complete-evidence rank: {rank}")
    print("  Top 3:")

    for position, item in enumerate(results[:3], start=1):
        document = item["document"]
        score = item.get(
            "rerank_score",
            item.get(
                "fusion_score",
                item.get("hybrid_score", item.get("score", 0.0)),
            ),
        )

        print(
            f"    {position}. chunk={document.get('chunk_id')} "
            f"| score={score:.4f} "
            f"| source={Path(document.get('source', '')).name}"
        )

    return metrics


def main():
    test_cases = load_test_cases()
    vector_store = VectorStore.load(str(PROJECT_ROOT / "storage"))
    retriever = HybridRetriever(vector_store)

    print(f"Loading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME)

    aggregate = {
        "Hybrid Only": [],
        "Reranker Only": [],
        "RRF": [],
    }

    for case in test_cases:
        question = case["question"]

        query_vector = create_embedding(question)
        hybrid = retriever.search(
            question=question,
            query_vector=query_vector,
            k=RETRIEVAL_K,
        )

        pairs = [
            (question, item["document"]["text"])
            for item in hybrid
        ]

        scores = (
            model.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
            )
            if pairs
            else []
        )

        reranked = [
            {
                **item,
                "rerank_score": float(score),
            }
            for item, score in zip(hybrid, scores)
        ]
        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        fused = reciprocal_rank_fusion(hybrid, reranked)

        print("\n" + "=" * 75)
        print(f"QUESTION: {question}")
        print(f"Expected source: {case['expected_source']}")
        print(f"Required evidence groups: {case['expected_evidence']}")

        for name, results in [
            ("Hybrid Only", hybrid),
            ("Reranker Only", reranked),
            ("RRF", fused),
        ]:
            metrics = evaluate_strategy(name, results, case)
            aggregate[name].append(metrics)

    print("\n" + "=" * 75)
    print("AGGREGATE COMPARISON")
    print("=" * 75)

    for name, results in aggregate.items():
        print(f"\n{name}")

        for metric in ("hit@3", "hit@10", "mrr"):
            average = (
                sum(result[metric] for result in results) / len(results)
                if results
                else 0.0
            )
            print(f"  {metric}: {average:.4f}")


if __name__ == "__main__":
    main()
