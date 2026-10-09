from rag.embeddings import create_embedding


text = "RAG is Retrieval Augmented Generation"


vector = create_embedding(text)


print(vector)

print("Length:", len(vector))