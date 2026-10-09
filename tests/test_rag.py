from rag.vectorstore import VectorStore
from rag.embeddings import create_embedding
from rag.rag_service import RAGService

documents = [
    "Python is a programming language.",
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant documents and uses an LLM to answer questions.",
]


vectors = [create_embedding(doc) for doc in documents]


store = VectorStore(dimension=len(vectors[0]))


store.add(vectors, documents)


rag = RAGService(store)


answer = rag.ask("What is RAG?")


print(answer)
