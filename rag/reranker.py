
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, contexts: list, top_k: int = 3):

        if not contexts:
            return []

        pairs = [
            (question, item["document"]["text"])
            for item in contexts
        ]

        scores = self.model.predict(pairs)

        results = []

        for context, score in zip(contexts, scores):
            results.append({
                **context,
                "rerank_score": float(score),
            })

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return results[:top_k]
