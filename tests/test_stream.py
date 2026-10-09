from llm import stream_ai_response

messages = [{"role": "user", "content": "Hi how are you?"}]


for token in stream_ai_response(messages):
    print(token, end="", flush=True)
