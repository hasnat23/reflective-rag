"""reflective-rag: Corrective RAG and Self-RAG as LangGraph graphs, with a shared evaluation harness."""

from .config import Settings
from .graphs import PIPELINES, CorrectiveRAG, NaiveRAG, Pipeline, RAGResult, SelfRAG
from .retrieval import Corpus, WikipediaSearch

__version__ = "0.1.0"

__all__ = ["PIPELINES", "Corpus", "CorrectiveRAG", "NaiveRAG", "Pipeline", "RAGResult", "SelfRAG", "Settings", "WikipediaSearch"]
