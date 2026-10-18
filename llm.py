
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
BASE_URL = "https://api.groq.com/openai/v1"

_client = None


class LLMServiceError(Exception):
    """Represents a handled LLM provider error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def get_client() -> OpenAI:

    global _client

    if _client is None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise LLMServiceError(
                "configuration_error",
                "GROQ_API_KEY is not configured.",
            )

        _client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=30.0,
            max_retries=2,
        )

    return _client


def translate_error(exc: OpenAIError) -> LLMServiceError:

    if isinstance(exc, AuthenticationError):
        return LLMServiceError(
            "authentication_error",
            "LLM provider authentication failed.",
        )

    if isinstance(exc, RateLimitError):
        return LLMServiceError(
            "rate_limit",
            "LLM provider rate limit exceeded.",
        )

    if isinstance(exc, APITimeoutError):
        return LLMServiceError(
            "timeout",
            "LLM provider request timed out.",
        )

    if isinstance(exc, APIConnectionError):
        return LLMServiceError(
            "connection_error",
            "Could not connect to the LLM provider.",
        )

    if isinstance(exc, APIStatusError):
        return LLMServiceError(
            "provider_error",
            f"LLM provider returned HTTP {exc.status_code}.",
        )

    return LLMServiceError(
        "llm_error",
        "LLM provider request failed.",
    )


def get_ai_response(messages: list[dict[str, str]]) -> str:

    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        if not response.choices:
            raise LLMServiceError(
                "empty_response",
                "LLM provider returned no choices.",
            )

        content = response.choices[0].message.content

        if content is None:
            raise LLMServiceError(
                "empty_response",
                "LLM provider returned no text content.",
            )

        return content

    except LLMServiceError:
        raise

    except OpenAIError as exc:
        raise translate_error(exc) from exc


def stream_ai_response(messages: list[dict[str, str]]):

    try:
        stream = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in stream:

            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    yield content

    except LLMServiceError:
        raise

    except OpenAIError as exc:
        raise translate_error(exc) from exc
