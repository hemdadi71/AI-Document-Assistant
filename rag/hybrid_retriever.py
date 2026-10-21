
import re

from rank_bm25 import BM25Okapi
from rag.vectorstore import VectorStore


class HybridRetriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.documents = vector_store.documents

        self.tokenized_documents = [
            self._tokenize(document.get("text", ""))
            for document in self.documents
        ]

        self.bm25 = (
            BM25Okapi(self.tokenized_documents)
            if any(self.tokenized_documents)
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _document_key(document: dict) -> tuple:
        return (
            document.get("source"),
            document.get("chunk_id"),
            document.get("text", ""),
        )

    def search(
        self,
        question: str,
        query_vector: list[float],
        k: int = 30,
        rrf_k: int = 60,
    ) -> list[dict]:

        if k <= 0 or not self.documents:
            return []

        dense_results = self.vector_store.search(
            query_vector,
            k=k,
        )

        dense_map = {}

        for rank, result in enumerate(dense_results, start=1):
            document = result["document"]
            key = self._document_key(document)

            dense_map[key] = {
                "document": document,
                "score": result["score"],
                "rank": rank,
            }

        # 2. Sparse Retrieval: جست‌وجوی واژه‌محور با BM25
        bm25_map = {}

        query_tokens = self._tokenize(question)

        if self.bm25 is not None and query_tokens:

            scores = self.bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )

            rank = 0

            for index in ranked_indices:

                score = float(scores[index])

                # نتایج بدون امتیاز مثبت را وارد نمی‌کنیم.
                if score <= 0:
                    continue

                document = self.documents[index]
                key = self._document_key(document)

                rank += 1

                bm25_map[key] = {
                    "document": document,
                    "score": score,
                    "rank": rank,
                }

                if rank >= k:
                    break

        # 3. ادغام رتبه‌ها با Reciprocal Rank Fusion
        all_keys = set(dense_map) | set(bm25_map)
        results = []

        for key in all_keys:

            dense = dense_map.get(key)
            sparse = bm25_map.get(key)

            hybrid_score = 0.0

            if dense:
                hybrid_score += 1 / (
                    rrf_k + dense["rank"]
                )

            if sparse:
                hybrid_score += 1 / (
                    rrf_k + sparse["rank"]
                )

            document = (
                dense["document"]
                if dense
                else sparse["document"]
            )

            results.append({
                "document": document,
                "score": (
                    dense["score"] if dense else 0.0
                ),
                "dense_score": (
                    dense["score"] if dense else None
                ),
                "bm25_score": (
                    sparse["score"] if sparse else None
                ),
                "hybrid_score": hybrid_score,
                "dense_rank": (
                    dense["rank"] if dense else None
                ),
                "bm25_rank": (
                    sparse["rank"] if sparse else None
                ),
            })

        results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return results[:k]
