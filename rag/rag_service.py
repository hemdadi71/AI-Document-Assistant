
from rag.embeddings import create_embedding
from rag.prompt import build_rag_prompt
from rag.vectorstore import VectorStore
from rag.reranker import Reranker
from rag.hybrid_retriever import HybridRetriever

from llm import get_ai_response
from llm import stream_ai_response

import os


THRESHOLD = 0.1
RETRIEVAL_K = 30
FINAL_CONTEXTS_K = 3


class RAGService:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

        # Initialize retrieval components once
        self.hybrid_retriever = HybridRetriever(vector_store)
        self.reranker = Reranker()

    def _retrieve_contexts(self, question: str):

        query_vector = create_embedding(question)

        # 1. Hybrid retrieval: FAISS + BM25 + RRF
        candidates = self.hybrid_retriever.search(
            question=question,
            query_vector=query_vector,
            k=RETRIEVAL_K,
        )

        if not candidates:
            return []

        # 2. Check dense and lexical evidence separately
        dense_scores = [
            item["dense_score"]
            for item in candidates
            if item["dense_score"] is not None
        ]

        has_bm25_match = any(
            item["bm25_score"] is not None
            for item in candidates
        )

        best_dense_score = max(dense_scores, default=0.0)

        # Keep the existing threshold for dense-only results.
        # Do not compare RRF or BM25 scores with the FAISS threshold.
        if best_dense_score < THRESHOLD and not has_bm25_match:
            return []

        # 3. Rerank the hybrid candidates
        contexts = self.reranker.rerank(
            question,
            candidates,
            top_k=FINAL_CONTEXTS_K,
        )

        return contexts

    def ask(self, question: str):

        contexts = self._retrieve_contexts(question)

        if not contexts:
            return {
                "answer": "I don't know.",
                "sources": [],
            }

        prompt = build_rag_prompt(question, contexts)

        answer = get_ai_response([
            {"role": "user", "content": prompt}
        ])

        return {
            "answer": answer,
            "sources": [
                {
                    "source": os.path.basename(
                        context["document"]["source"]
                    ),
                    "chunk_id": context["document"]["chunk_id"],
                    "score": context["score"],
                    "dense_score": context["dense_score"],
                    "bm25_score": context["bm25_score"],
                    "hybrid_score": context["hybrid_score"],
                    "rerank_score": context["rerank_score"],
                }
                for context in contexts
            ],
        }

    def stream_answer(self, question: str):

        contexts = self._retrieve_contexts(question)

        if not contexts:
            return iter(["I don't know."])

        prompt = build_rag_prompt(question, contexts)

        return stream_ai_response([
            {"role": "user", "content": prompt}
        ])
