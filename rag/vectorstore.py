
import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, vectors, documents):

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array.")

        if vectors.shape[1] != self.dimension:
            raise ValueError("Vector dimension mismatch.")

        if len(vectors) != len(documents):
            raise ValueError(
                "Number of vectors must match number of documents."
            )

        if len(vectors) == 0:
            return

        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_vector, k=3):

        if k <= 0 or self.index.ntotal == 0:
            return []

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "FAISS index and documents are out of sync."
            )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        ).reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Query vector dimension mismatch.")

        faiss.normalize_L2(query_vector)

        search_k = min(k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            search_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0 or index >= len(self.documents):
                continue

            results.append({
                "document": self.documents[index],
                "score": float(score)
            })

        return results

    def save(self, path: str):

        os.makedirs(path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss")
        )

        with open(
            os.path.join(path, "documents.pkl"), "wb"
        ) as f:
            pickle.dump(self.documents, f)

    def add_documents(self, vectors, documents):

        self.add(vectors, documents)

    @classmethod
    def load(cls, path: str):

        index = faiss.read_index(
            os.path.join(path, "index.faiss")
        )

        with open(
            os.path.join(path, "documents.pkl"), "rb"
        ) as f:
            documents = pickle.load(f)

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS index and documents have different sizes."
            )

        store = cls(index.d)
        store.index = index
        store.documents = documents

        return store
