
import os

from sentence_transformers import CrossEncoder

from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.hybrid_retriever import HybridRetriever


SOURCE = "The-Little-Prince-for-Level-2-PDF.pdf"

RETRIEVAL_K = 100
RRF_K = 60
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

MODEL_NAMES = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
]

METRIC_NAMES = [
    "candidate_recall",
    "reranker_recall@3",
    "reranker_recall@10",
    "reranker_mrr",
    "fusion_recall@3",
    "fusion_recall@10",
    "fusion_mrr",
]


def document_key(document):
    """Create a stable key for identifying a document chunk."""

    return (
        os.path.basename(document.get("source", "")),
        document.get("chunk_id"),
    )


def recall_at(ranked_keys, expected_keys, k):
    """Calculate the fraction of expected chunks found in top-k."""

    retrieved = set(ranked_keys[:k])

    return len(expected_keys & retrieved) / len(expected_keys)


def calculate_mrr(ranked_keys, expected_keys):
    """Calculate reciprocal rank of the first relevant chunk."""

    for rank, key in enumerate(ranked_keys, start=1):

        if key in expected_keys:
            return 1.0 / rank

    return 0.0


def rank_fusion(candidates, reranked):
    """
    Combine Hybrid Retrieval ranks and Cross-Encoder ranks
    using Reciprocal Rank Fusion (RRF).
    """

    hybrid_ranks = {
        document_key(item["document"]): rank
        for rank, item in enumerate(candidates, start=1)
    }

    reranker_ranks = {
        document_key(item["document"]): rank
        for rank, item in enumerate(reranked, start=1)
    }

    results = []

    for item in candidates:

        key = document_key(item["document"])

        hybrid_rank = hybrid_ranks[key]
        reranker_rank = reranker_ranks[key]

        fusion_score = (
            1.0 / (RRF_K + hybrid_rank)
            + 1.0 / (RRF_K + reranker_rank)
        )

        results.append({
            "item": item,
            "hybrid_rank": hybrid_rank,
            "reranker_rank": reranker_rank,
            "fusion_score": fusion_score,
        })

    results.sort(
        key=lambda result: result["fusion_score"],
        reverse=True,
    )

    return results


def evaluate_model(model, retriever, model_name):

    metrics = {
        name: []
        for name in METRIC_NAMES
    }

    print("\n" + "=" * 75)
    print(f"MODEL: {model_name}")
    print("=" * 75)

    for case in TEST_CASES:

        question = case["question"]

        expected_keys = {
            (SOURCE, chunk_id)
            for chunk_id in case["expected_chunk_ids"]
        }

        # --------------------------------------------------
        # 1. Hybrid Retrieval
        # --------------------------------------------------

        query_vector = create_embedding(question)

        candidates = retriever.search(
            question=question,
            query_vector=query_vector,
            k=RETRIEVAL_K,
        )

        candidate_keys = {
            document_key(item["document"])
            for item in candidates
        }

        candidate_recall = (
            len(expected_keys & candidate_keys)
            / len(expected_keys)
        )

        # --------------------------------------------------
        # 2. Cross-Encoder Reranking
        # --------------------------------------------------

        pairs = [
            (question, item["document"]["text"])
            for item in candidates
        ]

        if pairs:

            scores = model.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
            )

            reranked = [
                {
                    **item,
                    "rerank_score": float(score),
                }
                for item, score in zip(candidates, scores)
            ]

            reranked.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

        else:
            reranked = []

        reranked_keys = [
            document_key(item["document"])
            for item in reranked
        ]

        reranker_recall_3 = recall_at(
            reranked_keys, expected_keys, 3
        )

        reranker_recall_10 = recall_at(
            reranked_keys, expected_keys, 10
        )

        reranker_mrr = calculate_mrr(
            reranked_keys, expected_keys
        )

        # --------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # --------------------------------------------------

        fused = rank_fusion(
            candidates,
            reranked,
        )

        fused_keys = [
            document_key(result["item"]["document"])
            for result in fused
        ]

        fusion_recall_3 = recall_at(
            fused_keys, expected_keys, 3
        )

        fusion_recall_10 = recall_at(
            fused_keys, expected_keys, 10
        )

        fusion_mrr = calculate_mrr(
            fused_keys, expected_keys
        )

        # --------------------------------------------------
        # 4. Store metrics
        # --------------------------------------------------

        current_metrics = {
            "candidate_recall": candidate_recall,
            "reranker_recall@3": reranker_recall_3,
            "reranker_recall@10": reranker_recall_10,
            "reranker_mrr": reranker_mrr,
            "fusion_recall@3": fusion_recall_3,
            "fusion_recall@10": fusion_recall_10,
            "fusion_mrr": fusion_mrr,
        }

        for name, value in current_metrics.items():
            metrics[name].append(value)

        # --------------------------------------------------
        # 5. Print per-question results
        # --------------------------------------------------

        print("\n" + "-" * 75)
        print(f"QUESTION: {question}")
        print(f"Expected chunks: {case['expected_chunk_ids']}")

        print(
            f"Candidate Recall: {candidate_recall:.3f}"
        )

        print("\nCross-Encoder:")
        print(
            f"  Recall@3: {reranker_recall_3:.3f} | "
            f"Recall@10: {reranker_recall_10:.3f} | "
            f"MRR: {reranker_mrr:.3f}"
        )

        print("  Expected chunk ranks:")

        for expected_key in sorted(expected_keys):

            if expected_key in reranked_keys:
                rank = reranked_keys.index(expected_key) + 1

                print(
                    f"    Chunk {expected_key[1]}: "
                    f"rank={rank}"
                )
            else:
                print(
                    f"    Chunk {expected_key[1]}: "
                    "not retrieved"
                )

        print("  Top 3:")

        for rank, item in enumerate(reranked[:3], start=1):

            print(
                f"    {rank}. "
                f"chunk={item['document']['chunk_id']} | "
                f"score={item['rerank_score']:.4f}"
            )

        print("\nRank Fusion (RRF):")
        print(
            f"  Recall@3: {fusion_recall_3:.3f} | "
            f"Recall@10: {fusion_recall_10:.3f} | "
            f"MRR: {fusion_mrr:.3f}"
        )

        print("  Expected chunk ranks:")

        for expected_key in sorted(expected_keys):

            if expected_key in fused_keys:
                rank = fused_keys.index(expected_key) + 1

                print(
                    f"    Chunk {expected_key[1]}: "
                    f"rank={rank}"
                )
            else:
                print(
                    f"    Chunk {expected_key[1]}: "
                    "not retrieved"
                )

        print("  Top 3:")

        for rank, result in enumerate(fused[:3], start=1):

            document = result["item"]["document"]

            print(
                f"    {rank}. chunk={document['chunk_id']} | "
                f"hybrid_rank={result['hybrid_rank']} | "
                f"reranker_rank={result['reranker_rank']} | "
                f"fusion_score={result['fusion_score']:.6f}"
            )

    return metrics


def print_aggregate_results(all_results):

    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)

    for metric_name in METRIC_NAMES:

        print(f"\n{metric_name}")

        for model_name in MODEL_NAMES:

            values = all_results[model_name][metric_name]

            average = (
                sum(values) / len(values)
                if values
                else 0.0
            )

            print(f"  {model_name}: {average:.4f}")


def main():

    vector_store = VectorStore.load("storage")

    retriever = HybridRetriever(vector_store)

    all_results = {}

    for model_name in MODEL_NAMES:

        print(f"\nLoading model: {model_name}")

        model = CrossEncoder(model_name)

        all_results[model_name] = evaluate_model(
            model=model,
            retriever=retriever,
            model_name=model_name,
        )

        del model

    print_aggregate_results(all_results)


if __name__ == "__main__":
    main()
