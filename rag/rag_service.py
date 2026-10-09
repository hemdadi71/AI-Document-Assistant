from rag.embeddings import create_embedding
from rag.prompt import build_rag_prompt
from rag.vectorstore import VectorStore
from llm import get_ai_response
from llm import stream_ai_response
import os

class RAGService:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def ask(self, question: str) -> str:

        query_vector = create_embedding(question)

        contexts = self.vector_store.search(query_vector, k=1)

        prompt = build_rag_prompt(question, contexts)

        answer = get_ai_response([{"role": "user", "content": prompt}])

        return {
            "answer": answer,
            "sources": [
                {"source": os.path.basename(context["source"]), "chunk_id": context["chunk_id"]}
                for context in contexts
            ],
        }

    def stream_answer(self, question: str):

        query_vector = create_embedding(question)

        contexts = self.vector_store.search(query_vector, k=1)

        prompt = build_rag_prompt(question, contexts)

        return stream_ai_response([{"role": "user", "content": prompt}])
