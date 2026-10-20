
import pickle

TARGET_CHUNKS = {
    7, 8,                  # Draw a sheep
    81, 85, 84,             # Man on the second planet
    72, 77, 80,             # The king
    154, 73, 83,            # Fifty-three minutes
}

with open("storage/documents.pkl", "rb") as file:
    documents = pickle.load(file)

for document in documents:
    chunk_id = document.get("chunk_id")

    if chunk_id in TARGET_CHUNKS:
        print(f"\n{'=' * 60}")
        print(f"CHUNK ID: {chunk_id}")
        print(document.get("text", ""))
