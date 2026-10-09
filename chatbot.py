from typing import List, Dict

from llm import get_ai_response

Message = Dict[str, str]


class ChatBot:

    def __init__(self):
        self.history: List[Message] = []

    def add_message(self, role: str, content: str) -> None:

        self.history.append({"role": role, "content": content})

    def chat(self, question: str) -> str:

        self.add_message("user", question)

        answer = get_ai_response(self.history)

        self.add_message("assistant", answer)

        return answer
