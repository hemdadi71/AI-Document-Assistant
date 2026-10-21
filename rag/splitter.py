
import re


def split_text(
    text: str,
    source: str = None,
    chunk_size: int = 1000,
    overlap: int = 200,
):
    if not text or not text.strip():
        return []

    # Normalize line endings and preserve paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = re.split(r"\n\s*\n+", text)

    units = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        # Split paragraphs into sentences where possible.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Keep normal sentences intact.
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue

            # Split unusually long sentences by words.
            words = sentence.split()
            current_words = []

            for word in words:
                candidate = " ".join(current_words + [word])

                if current_words and len(candidate) > chunk_size:
                    units.append(" ".join(current_words))

                    # Preserve a small overlap for long sentences.
                    tail = []
                    tail_length = 0

                    for previous_word in reversed(current_words):
                        extra = len(previous_word) + (
                            1 if tail else 0
                        )

                        if tail_length + extra > overlap:
                            break

                        tail.insert(0, previous_word)
                        tail_length += extra

                    current_words = tail

                current_words.append(word)

            if current_words:
                units.append(" ".join(current_words))

    chunks = []
    current_units = []
    current_length = 0

    for unit in units:
        extra_length = len(unit) + (
            1 if current_units else 0
        )

        if (
            current_units
            and current_length + extra_length > chunk_size
        ):
            chunks.append(" ".join(current_units))

            # Keep complete previous sentences as overlap.
            overlap_units = []
            overlap_length = 0

            for previous_unit in reversed(current_units):
                extra = len(previous_unit) + (
                    1 if overlap_units else 0
                )

                if overlap_length + extra > overlap:
                    break

                overlap_units.insert(0, previous_unit)
                overlap_length += extra

            current_units = overlap_units
            current_length = sum(
                len(item) for item in current_units
            ) + max(0, len(current_units) - 1)

            # Ensure the next unit fits.
            while (
                current_units
                and current_length + 1 + len(unit) > chunk_size
            ):
                removed = current_units.pop(0)
                current_length -= len(removed)

                if current_units:
                    current_length -= 1

        if current_units:
            current_length += 1

        current_units.append(unit)
        current_length += len(unit)

    if current_units:
        chunks.append(" ".join(current_units))

    return [
        {
            "text": chunk,
            "source": source,
            "chunk_id": chunk_id,
        }
        for chunk_id, chunk in enumerate(chunks)
    ]
