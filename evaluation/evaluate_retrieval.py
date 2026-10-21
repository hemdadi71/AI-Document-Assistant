
import os
import re

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
        "evidence": ["snake who was eating a big animal"],
    },
    {
        "question": "How many sunsets did the little prince watch in one day?",
        "evidence": ["forty-three times"],
    },
    {
        "question": "What did the little prince ask the narrator to draw?",
        "evidence": [
            "please, draw me a sheep",
            "draw me a sheep",
        ],
    },
    {
        "question": "What did the little prince's flower ask him to bring?",
        "evidence": [
            "could you bring me some water",
            "bring me some water",
        ],
    },
    {
        "question": "Why did the man on the second planet want people to admire him?",
        "evidence": [
            "most handsome",
            "best dressed",
            "most intelligent",
        ],
    },
    {
        "question": "What did the king say he ruled over?",
        "evidence": [
            "i control everything",
            "king of all the universe",
        ],
    },
    {
        "question": "What would the little prince do with fifty-three extra minutes?",
        "evidence": ["walk very slowly to a fountain"],
    },
    {
        "question": "Why was the narrator worried about the little prince's flower at the end of the story?",
        "evidence": [
            "sheep can come to the flower",
            "glass over the flower",
        ],
    },
]


def normalize(text: str) -> str:
    """Normalize whitespace and letter casing."""

    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_evidence(item, evidence_phrases):
    """Check whether a candidate contains expected evidence."""

    document = item["document"]

    if os.path.basename(document.get("source", "")) != SOURCE:
        return False

    text = normalize(document.get("text", ""))

    return any(
        normalize(phrase) in text
        for phrase in evidence_phrases
    )


def evidence_rank(results, evidence_phrases):
    """Return the rank of the first relevant result."""

    for rank, item in enumerate(results, start=1):
        if contains_evidence(item, evidence_phrases):
            return rank

    return None


def hit_at(results, evidence_phrases, k):
    """Return 1 if expected evidence exists in the top-k."""

    return int(
        evidence_rank(results[:k], evidence_phrases)
        is not None
    )


def reciprocal_rank(results, evidence_phrases):
    rank = evidence_rank(results, evidence_phrases)

    return 1.0 / rank if rank is not None else 0.0


def evaluate_question(case, retriever, model):

    question = case["question"]
    evidence = case["evidence"]

    query_vector = create_embedding(question)

    candidates = retriever.search(
        question=question,
        query_vector=query_vector,
        k=RETRIEVAL_K,
    )

    # Evaluate Hybrid Retrieval before reranking.
    retrieval_hit_30 = hit_at(candidates, evidence, 30)
    retrieval_hit_100 = hit_at(candidates, evidence, 100)

    # Score every retrieved candidate using the Cross-Encoder.
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

    # Evaluate the final ranking.
    rerank_hit_3 = hit_at(reranked, evidence, 3)
    rerank_hit_10 = hit_at(reranked, evidence, 10)
    rerank_mrr = reciprocal_rank(reranked, evidence)

    print("\n" + "=" * 75)
    print(f"QUESTION: {question}")

    print("\nHYBRID RETRIEVAL")
    print(f"Candidates: {len(candidates)}")
    print(f"Evidence Hit@30: {retrieval_hit_30}")
    print(f"Evidence Hit@100: {retrieval_hit_100}")

    retrieval_rank = evidence_rank(candidates, evidence)

    print(f"First evidence rank: {retrieval_rank}")

    print("\nCROSS-ENCODER RERANKING")
    print(f"Evidence Hit@3: {rerank_hit_3}")
    print(f"Evidence Hit@10: {rerank_hit_10}")
    print(f"MRR: {rerank_mrr:.4f}")

    rerank_rank = evidence_rank(reranked, evidence)

    print(f"First evidence rank: {rerank_rank}")

    print("\nTOP 3 RERANKED RESULTS")

    for rank, item in enumerate(reranked[:3], start=1):
        document = item["document"]

        print(
            f"{rank}. chunk={document['chunk_id']} | "
            f"score={item['rerank_score']:.4f} | "
            f"source={os.path.basename(document['source'])}"
        )

        if contains_evidence(item, evidence):
            print("   Expected evidence found.")

    return {
        "retrieval_hit@30": retrieval_hit_30,
        "retrieval_hit@100": retrieval_hit_100,
        "rerank_hit@3": rerank_hit_3,
        "rerank_hit@10": rerank_hit_10,
        "rerank_mrr": rerank_mrr,
    }


def main():

    vector_store = VectorStore.load("storage")
    retriever = HybridRetriever(vector_store)

    print(f"Loading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME)

    all_results = []

    for case in TEST_CASES:
        result = evaluate_question(
            case=case,
            retriever=retriever,
            model=model,
        )

        all_results.append(result)

    print("\n" + "=" * 75)
    print("AGGREGATE EVALUATION")
    print("=" * 75)

    metric_names = [
        "retrieval_hit@30",
        "retrieval_hit@100",
        "rerank_hit@3",
        "rerank_hit@10",
        "rerank_mrr",
    ]

    for metric in metric_names:
        average = sum(
            result[metric]
            for result in all_results
        ) / len(all_results)

        print(f"{metric}: {average:.4f}")


if __name__ == "__main__":
    main()
