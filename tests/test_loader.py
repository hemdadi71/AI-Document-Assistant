from rag.loader import load_pdf


text = load_pdf("data/pdf/sample.pdf")


print(text[:1000])