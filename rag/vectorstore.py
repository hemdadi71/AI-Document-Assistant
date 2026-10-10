import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(self, vectors, documents):

        vectors = np.array(vectors).astype("float32")

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        self.documents.extend(documents)

    def search(self, query_vector, k=3):

        query_vector = np.array([query_vector]).astype("float32")

        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        results = []

        for index in indices[0]:

            results.append(self.documents[index])

        return results

    def save(self, path: str):

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/documents.pkl", "wb") as f:

            pickle.dump(self.documents, f)

    def add_documents(self, vectors, documents):

        self.index.add(np.array(vectors).astype("float32"))

        self.documents.extend(documents)

    @classmethod
    def load(cls, path: str):

        index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/documents.pkl", "rb") as f:

            documents = pickle.load(f)

        store = cls(index.d)

        store.index = index

        store.documents = documents

        return store
