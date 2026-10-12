"""Model factories and an LLM call counter."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    return init_chat_model(settings.model, model_provider=settings.provider, temperature=0)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embedding_model)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai

        return OpenAIEmbeddings(model=settings.embedding_model)
    raise ValueError(f"unsupported embedding provider {settings.embedding_provider!r}")


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls and tokens across a graph run."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self.calls += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
