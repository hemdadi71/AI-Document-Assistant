def build_rag_prompt(question, contexts):

    context_text = "\n\n".join([f"""
Source: {item['document']['source']}
Score: {item['score']}

{item['document']['text']}
""" for item in contexts])

    prompt = f"""
You are an AI assistant. Answer the question using only the provided context.

Context:

{context_text}

Question:
{question}

If the answer is not available in the context, say:
"I don't know."
"""

    return prompt
