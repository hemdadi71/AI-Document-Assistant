def build_rag_prompt(question, contexts):

    context_text = "\n\n".join([context["text"] for context in contexts])

    prompt = f"""
You are an AI assistant.

Answer the question only based on the provided context.

Context:

{context_text}


Question:

{question}

Answer:
"""

    return prompt
