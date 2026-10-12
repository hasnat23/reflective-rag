from .base import Pipeline, RAGResult, RAGState
from .crag import CorrectiveRAG
from .naive import NaiveRAG
from .self_rag import SelfRAG

PIPELINES: dict[str, type[Pipeline]] = {"naive": NaiveRAG, "crag": CorrectiveRAG, "self_rag": SelfRAG}

__all__ = ["PIPELINES", "CorrectiveRAG", "NaiveRAG", "Pipeline", "RAGResult", "RAGState", "SelfRAG"]
