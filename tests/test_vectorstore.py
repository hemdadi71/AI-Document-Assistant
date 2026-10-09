from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore

documents = [
    "Python is a programming language",
    "RAG uses retrieval augmented generation",
    "Employees have 30 days vacation",
]


# ساخت embedding برای هر متن
vectors = [create_embedding(doc) for doc in documents]


# ساخت Vector Store
store = VectorStore(dimension=len(vectors[0]))


# اضافه کردن داده‌ها
store.add(vectors, documents)


# سوال کاربر
question = "What is Retrieval Augmented Generation?"


# تبدیل سوال به embedding
query_vector = create_embedding(question)


# جستجو
results = store.search(query_vector, k=2)


print("Search results:")

for result in results:
    print("- ", result)
