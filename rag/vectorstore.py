
import faiss
import numpy as np
import pickle
import os
import tempfile
import warnings


class VectorStore:
    SNAPSHOT_FILENAME = "vectorstore.snapshot"
    INDEX_FILENAME = "index.faiss"
    DOCUMENTS_FILENAME = "documents.pkl"
    SNAPSHOT_VERSION = 1

    def __init__(self, dimension):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

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

    def _validate_sync(self):
        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "FAISS index and documents are out of sync."
            )

    @staticmethod
    def _atomic_write(path, data):
        """
        Write bytes to a temporary file and atomically replace
        the destination once the temporary file is complete.
        """
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)

        fd, temporary_path = tempfile.mkstemp(
            prefix=".vectorstore-",
            suffix=".tmp",
            dir=directory,
        )

        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, path)
            temporary_path = None

            # Best-effort directory metadata synchronization.
            try:
                directory_fd = os.open(
                    directory,
                    os.O_RDONLY,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass

        finally:
            if (
                temporary_path is not None
                and os.path.exists(temporary_path)
            ):
                os.unlink(temporary_path)

    @classmethod
    def exists(cls, path):
        """
        Detect either a new snapshot or legacy storage artifacts.
        A partially present legacy store is detected so that
        load() can report the underlying problem.
        """
        snapshot_path = os.path.join(
            path,
            cls.SNAPSHOT_FILENAME,
        )

        if os.path.isfile(snapshot_path):
            return True

        return any(
            os.path.exists(os.path.join(path, filename))
            for filename in (
                cls.INDEX_FILENAME,
                cls.DOCUMENTS_FILENAME,
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

        self._validate_sync()

        # Avoid modifying the caller's array.
        vectors = vectors.copy()
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(self, query_vector, k=3):
        if k <= 0 or self.index.ntotal == 0:
            return []

        self._validate_sync()

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Query vector dimension mismatch.")

        query_vector = query_vector.copy()
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
        self._validate_sync()

        sources = {}

        for document in self.documents:
            source = document.get("source")

            if not isinstance(source, str) or not source:
                continue

            sources[source] = sources.get(source, 0) + 1

        return sources

    def remove_source(self, source):
        """Remove every indexed chunk belonging to one source."""
        self._validate_sync()

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

        all_vectors = self.index.reconstruct_n(
            0,
            self.index.ntotal,
        )

        remaining_documents = [
            self.documents[i]
            for i in keep_indices
        ]

        new_index = faiss.IndexFlatIP(self.dimension)

        if keep_indices:
            remaining_vectors = np.asarray(
                all_vectors[keep_indices],
                dtype=np.float32,
            ).copy()

            new_index.add(remaining_vectors)

        # Update in-memory state only after rebuilding succeeds.
        self.index = new_index
        self.documents = remaining_documents

        return removed_count

    def save(self, path: str):
        """
        Persist the complete store in one atomic snapshot.

        Legacy FAISS and pickle files are maintained as compatibility
        mirrors. The snapshot is authoritative whenever it exists.
        """
        os.makedirs(path, exist_ok=True)
        self._validate_sync()

        # Serialize both components before modifying persisted state.
        index_bytes = faiss.serialize_index(
            self.index
        ).tobytes()

        documents_bytes = pickle.dumps(
            self.documents,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        snapshot_bytes = pickle.dumps(
            {
                "format_version": self.SNAPSHOT_VERSION,
                "index_bytes": index_bytes,
                "documents": self.documents,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        snapshot_path = os.path.join(
            path,
            self.SNAPSHOT_FILENAME,
        )

        # Commit point: one atomic replacement for the complete store.
        self._atomic_write(
            snapshot_path,
            snapshot_bytes,
        )

        # Maintain legacy files for compatibility. These are not
        # authoritative, so a mirror failure must not invalidate
        # the successfully committed snapshot.
        try:
            self._atomic_write(
                os.path.join(path, self.INDEX_FILENAME),
                index_bytes,
            )

            self._atomic_write(
                os.path.join(path, self.DOCUMENTS_FILENAME),
                documents_bytes,
            )

        except OSError as exc:
            warnings.warn(
                "Snapshot was saved successfully, but a legacy "
                f"storage mirror could not be updated: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def add_documents(self, vectors, documents):
        self.add(vectors, documents)

    @classmethod
    def load(cls, path: str):
        snapshot_path = os.path.join(
            path,
            cls.SNAPSHOT_FILENAME,
        )

        if os.path.isfile(snapshot_path):
            # Do not silently fall back to potentially stale legacy
            # files if an existing authoritative snapshot is corrupt.
            try:
                with open(snapshot_path, "rb") as file:
                    snapshot = pickle.load(file)

                if not isinstance(snapshot, dict):
                    raise ValueError("Invalid snapshot format.")

                if (
                    snapshot.get("format_version")
                    != cls.SNAPSHOT_VERSION
                ):
                    raise ValueError(
                        "Unsupported snapshot version."
                    )

                index_bytes = snapshot.get("index_bytes")
                documents = snapshot.get("documents")

                if not isinstance(index_bytes, bytes):
                    raise ValueError(
                        "Snapshot index data is invalid."
                    )

                if not isinstance(documents, list):
                    raise ValueError(
                        "Snapshot documents are invalid."
                    )

                index = faiss.deserialize_index(
                    np.frombuffer(
                        index_bytes,
                        dtype=np.uint8,
                    )
                )

            except Exception as exc:
                raise ValueError(
                    f"Unable to load vector store snapshot: {exc}"
                ) from exc

        else:
            # Backward compatibility with existing installations.
            index_path = os.path.join(
                path,
                cls.INDEX_FILENAME,
            )
            documents_path = os.path.join(
                path,
                cls.DOCUMENTS_FILENAME,
            )

            index = faiss.read_index(index_path)

            with open(documents_path, "rb") as file:
                documents = pickle.load(file)

        if not isinstance(documents, list):
            raise ValueError(
                "Vector store documents must be a list."
            )

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS index and documents have different sizes."
            )

        store = cls(index.d)
        store.index = index
        store.documents = documents

        return store
