from rag.loader import load_pdf
from rag.splitter import split_text

text = load_pdf("data/pdf/sample.pdf")


chunks = split_text(text)


print("Number of chunks:", len(chunks))


print("\nFirst chunk:")
print(chunks[0])
