
import pickle

TARGET_IDS = {
    7, 8, 15, 150, 192, 86, 71,
    193, 44, 189, 58, 43, 45,
}

with open("storage/documents.pkl", "rb") as file:
    documents = pickle.load(file)

for doc in documents:
    chunk_id = doc.get("chunk_id")

    if chunk_id not in TARGET_IDS:
        continue

    print("\n" + "=" * 70)
    print(f"CHUNK ID: {chunk_id}")
    print(doc.get("text", ""))
