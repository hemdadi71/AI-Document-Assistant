from rag.embeddings import create_embedding
from rag.prompt import build_rag_prompt
from rag.vectorstore import VectorStore
from rag.reranker import Reranker

from llm import get_ai_response
from llm import stream_ai_response

import os

THRESHOLD = 0.1
RETRIEVAL_K = 30
FINAL_CONTEXTS_K = 3


class RAGService:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.reranker = Reranker()

    def _retrieve_contexts(self, question: str):

        query_vector = create_embedding(question)

        candidates = self.vector_store.search(
            query_vector,
            k=RETRIEVAL_K,
        )

        if not candidates or candidates[0]["score"] < THRESHOLD:
            return []

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
