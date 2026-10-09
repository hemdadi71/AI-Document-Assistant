from llm import get_ai_response

messages = [{"role": "user", "content": "Explain RAG"}]


answer = get_ai_response(messages)

print(answer)
