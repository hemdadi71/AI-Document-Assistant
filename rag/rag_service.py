from rag.embeddings import create_embedding
from rag.prompt import build_rag_prompt
from rag.vectorstore import VectorStore
from llm import get_ai_response
from llm import stream_ai_response
import os

THRESHOLD = 0.1

class RAGService:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def ask(self, question: str) -> str:

            query_vector = create_embedding(question)

            contexts = self.vector_store.search(query_vector, k=3)

            if not contexts or contexts[0]["score"] < THRESHOLD:
                return {
            "answer": "I don't know.",
            "sources": []
            }

            prompt = build_rag_prompt(question, contexts)

            answer = get_ai_response([{"role": "user", "content": prompt}])

            return {
        "answer": answer,
        "sources": [
        {
        "source": os.path.basename(context["document"]["source"]),
        "chunk_id": context["document"]["chunk_id"],
        "score": context["score"],
        }
        for context in contexts
        ],
        }

    def stream_answer(self, question: str):

            query_vector = create_embedding(question)

            contexts = self.vector_store.search(query_vector, k=3)

            prompt = build_rag_prompt(question, contexts)

            return stream_ai_response([{"role": "user", "content": prompt}])
