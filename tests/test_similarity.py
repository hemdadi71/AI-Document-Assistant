from rag.embeddings import create_embedding
from rag.similarity import find_best_match

chunks = [
    "Python is a programming language",
    "RAG is Retrieval Augmented Generation",
    "Employees have 30 days vacation",
]


chunk_vectors = [create_embedding(chunk) for chunk in chunks]


question = "Explain RAG"


query_vector = create_embedding(question)


result, score = find_best_match(query_vector, chunk_vectors, chunks)


print("Best chunk:")
print(result)

print("Score:")
print(score)
