
def build_rag_prompt(question, contexts):

    context_blocks = []

    for i, item in enumerate(contexts, start=1):

        document = item["document"]

        context_blocks.append(
            f"""
Context {i}
Source: {document.get("source", "Unknown")}
Chunk ID: {document.get("chunk_id", "Unknown")}

{document["text"]}
"""
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are a precise question-answering assistant.

Answer the user's question using only the provided context.

Instructions:
1. Answer the exact question being asked.
2. Prefer explicit evidence that directly answers the question.
3. Contexts may discuss similar topics but describe different
   events, situations, places, or hypothetical scenarios.
   Do not confuse them.
4. When multiple contexts contain related facts, select the
   fact that matches the exact subject and situation in the question.
5. Do not choose an answer merely because a context mentions
   similar keywords.
6. Do not invent facts or use outside knowledge.
7. If the answer cannot be determined from the context,
   respond exactly with: I don't know.
8. Keep the answer concise.

Context:

{context_text}

Question:
{question}

Answer:
"""

    return prompt
