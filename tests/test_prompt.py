from rag.prompt import build_rag_prompt

question = "What is RAG?"


contexts = [
    "RAG stands for Retrieval Augmented Generation.",
    "RAG retrieves relevant information and uses an LLM to generate answers.",
]


prompt = build_rag_prompt(question, contexts)


print(prompt)
