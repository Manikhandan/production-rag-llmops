from __future__ import annotations

from typing import Protocol

from ragops.generate import ExtractiveProvider, estimate_tokens
from ragops.prompts import render
from ragops.types import Answer, Hit


class LLMProvider(Protocol):
    name: str

    def generate(
        self, question: str, hits: list[Hit], prompt_version: str, max_context_chars: int
    ) -> Answer: ...


class ProviderError(RuntimeError):
    pass


class HttpLLMProvider:
    """Adapter for a remote chat API. Not invoked in default CI.

    Configure RAG_PROVIDER=http, RAG_LLM_URL, and RAG_LLM_API_KEY.
    This class constructs the request contract; it does not run billed calls
    unless those settings are present and generate() is actually invoked.
    """

    name = "http"

    def __init__(self, url: str, api_key: str, timeout_s: float) -> None:
        if not url:
            raise ProviderError("RAG_LLM_URL is required for provider=http")
        if not api_key:
            raise ProviderError("RAG_LLM_API_KEY is required for provider=http")
        self.url = url
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate(
        self, question: str, hits: list[Hit], prompt_version: str, max_context_chars: int
    ) -> Answer:
        prompt = render(prompt_version, question, "")
        raise ProviderError(
            f"http provider is configured ({self.url}) but remote calls are disabled "
            f"in this repository; use extractive for local/CI. prompt_tokens~{estimate_tokens(prompt)}"
        )


def get_provider(
    name: str, *, llm_url: str = "", llm_api_key: str = "", timeout_s: float = 8.0
) -> LLMProvider:
    if name == "extractive":
        return ExtractiveProvider()
    if name == "http":
        return HttpLLMProvider(llm_url, llm_api_key, timeout_s)
    raise ValueError(f"unknown provider {name}")
