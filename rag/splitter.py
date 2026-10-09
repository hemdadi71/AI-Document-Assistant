def split_text(text: str, source: str = None):

    chunks = []

    chunk_size = 500
    overlap = 100

    start = 0
    chunk_id = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({"text": chunk_text, "source": source, "chunk_id": chunk_id})

        chunk_id += 1

        start += chunk_size - overlap

    return chunks
