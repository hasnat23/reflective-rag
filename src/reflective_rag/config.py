"""Runtime configuration, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Chat model. Any provider supported by langchain's init_chat_model works:
    # ollama (default, local), openai, anthropic, groq, google_genai, ...
    provider: str = field(default_factory=lambda: _env("RRAG_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: _env("RRAG_MODEL", "llama3.2:3b"))
    # Embeddings: "ollama" or "openai".
    embedding_provider: str = field(default_factory=lambda: _env("RRAG_EMBEDDING_PROVIDER", "ollama"))
    embedding_model: str = field(default_factory=lambda: _env("RRAG_EMBEDDING_MODEL", "nomic-embed-text"))

    top_k: int = 4
    # CRAG retrieval-evaluator thresholds on a 1-5 relevance scale.
    upper_threshold: int = 4
    lower_threshold: int = 2
    web_results: int = 3
    # Self-RAG loop limits.
    max_retrievals: int = 2
    max_generations: int = 2
    # "short" asks for a few-word answer (for EM/F1 evaluation), "full" for prose.
    answer_style: str = "full"
