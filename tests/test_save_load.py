from rag.vectorstore import VectorStore
from rag.embeddings import create_embedding

documents = [
    "Python is a programming language",
    "RAG uses retrieval augmented generation",
    "Employees have 30 days vacation",
]


vectors = [create_embedding(doc) for doc in documents]


store = VectorStore(len(vectors[0]))


store.add(vectors, documents)


# save
store.save("storage")


print("Saved successfully")


# load
new_store = VectorStore.load("storage")


print("Loaded successfully")


query = create_embedding("What is RAG?")


result = new_store.search(query, k=1)


print(result)
