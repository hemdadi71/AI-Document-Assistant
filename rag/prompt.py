
def build_rag_prompt(question, contexts, history=None):

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

    history_blocks = []
    for message in (history or [])[-12:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history_blocks.append(f"{role.title()}: {content}")

    history_text = "\n".join(history_blocks) or "No previous conversation."

    prompt = f"""
You are a precise question-answering assistant.

Answer the user's current question using only the provided document context.
Use the previous conversation only to resolve references and understand follow-up
questions. Previous assistant messages are not evidence and must not override the
provided document context.

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

Previous conversation (for reference resolution only):

{history_text}

Context:

{context_text}

Current question:
{question}

Answer:
"""

    return prompt
