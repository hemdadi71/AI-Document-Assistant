
import pickle
import os

with open("storage/documents.pkl", "rb") as f:
    documents = pickle.load(f)

searches = {
    3: ["draw me a sheep", "need a sheep"],
    4: ["bring me", "some water", "water"],
    5: ["admire me", "handsome", "best dressed", "richest"],
    6: ["I rule over", "all the stars", "universe"],
    7: ["fifty-three minutes", "fifty three minutes", "fountain"],
    8: ["glass", "sheep", "flower", "muzzle"],
}

for question_number, keywords in searches.items():

    print(f"\n{'=' * 70}")
    print(f"QUESTION {question_number}")

    for doc in documents:

        if os.path.basename(doc.get("source", "")) != (
            "The-Little-Prince-for-Level-2-PDF.pdf"
        ):
            continue

        text = doc.get("text", "")
        normalized = text.lower()

        matches = [
            keyword
            for keyword in keywords
            if keyword.lower() in normalized
        ]

        if matches:
            print(
                f"\nChunk ID: {doc.get('chunk_id')}"
                f"\nMatched: {matches}"
                f"\n{text[:700]}"
            )
