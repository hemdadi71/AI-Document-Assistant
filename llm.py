from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)


MODEL_NAME = "openai/gpt-oss-20b"


def get_ai_response(messages: list[dict[str, str]]) -> str:

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    return response.choices[0].message.content


def stream_ai_response(messages: list[dict[str, str]]):

    stream = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, stream=True
    )
    for chunk in stream:

        if chunk.choices[0].delta.content:

            yield chunk.choices[0].delta.content
