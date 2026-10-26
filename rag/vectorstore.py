
import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    @staticmethod
    def _normalize_source(source):
        if not isinstance(source, str) or not source.strip():
            return None

        return os.path.normcase(
            os.path.abspath(
                os.path.normpath(source)
            )
        )

    def add(self, vectors, documents):
        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

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
            dtype=np.float32,
        ).reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Query vector dimension mismatch.")

        faiss.normalize_L2(query_vector)

        search_k = min(k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            search_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.documents):
                continue

            results.append({
                "document": self.documents[index],
                "score": float(score),
            })

        return results

    def list_sources(self):
        """Return the number of indexed chunks per source."""

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "FAISS index and documents are out of sync."
            )

        sources = {}

        for document in self.documents:
            source = document.get("source")

            if not isinstance(source, str) or not source:
                continue

            sources[source] = sources.get(source, 0) + 1

        return sources

    def remove_source(self, source):
        """Remove every indexed chunk belonging to one source."""

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "FAISS index and documents are out of sync."
            )

        target = self._normalize_source(source)

        if target is None:
            raise ValueError("Source path is required.")

        keep_indices = [
            i
            for i, document in enumerate(self.documents)
            if self._normalize_source(
                document.get("source")
            ) != target
        ]

        removed_count = len(self.documents) - len(keep_indices)

        if removed_count == 0:
            return 0

        # Recover the vectors for the chunks we want to keep.
        all_vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal,
        )

        remaining_documents = [
            self.documents[i]
            for i in keep_indices
        ]

        # Rebuild the index so vector positions match document positions.
        new_index = faiss.IndexFlatIP(self.dimension)

        if keep_indices:
            remaining_vectors = np.asarray(
                all_vectors[keep_indices],
                dtype=np.float32,
            )

            new_index.add(remaining_vectors)

        self.index = new_index
        self.documents = remaining_documents

        return removed_count

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "Cannot save: FAISS index and documents are out of sync."
            )

        faiss.write_index(
            self.index,
            os.path.join(path, "index.faiss"),
        )

        with open(
            os.path.join(path, "documents.pkl"),
            "wb",
        ) as file:
            pickle.dump(self.documents, file)

    def add_documents(self, vectors, documents):
        self.add(vectors, documents)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(
            os.path.join(path, "index.faiss"),
        )

        with open(
            os.path.join(path, "documents.pkl"),
            "rb",
        ) as file:
            documents = pickle.load(file)

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS index and documents have different sizes."
            )

        store = cls(index.d)
        store.index = index
        store.documents = documents

        return store
