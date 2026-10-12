"""Shared graph state and the pipeline wrapper."""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from ..config import Settings
from ..graders import Graders
from ..llm import CallCounter
from ..retrieval import Corpus, WikipediaSearch


class RAGState(TypedDict, total=False):
    question: str
    query: str
    documents: list[Document]
    generation: str
    action: str  # CRAG: correct / incorrect / ambiguous
    retrievals: int
    generations: int
    grounded: bool | None  # Self-RAG [IsSup]
    useful: bool | None  # Self-RAG [IsUse]
    trace: Annotated[list[str], operator.add]


@dataclass
class RAGResult:
    question: str
    answer: str
    documents: list[Document]
    trace: list[str]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Base class: subclasses build a compiled LangGraph in ``build()``."""

    name = "base"

    def __init__(
        self,
        llm: BaseChatModel,
        corpus: Corpus,
        settings: Settings | None = None,
        web: WikipediaSearch | None = None,
    ) -> None:
        self.llm = llm
        self.corpus = corpus
        self.settings = settings or Settings()
        self.web = web if web is not None else WikipediaSearch()
        self.graders = Graders(llm, self.settings.answer_style)
        self.graph = self.build()

    def build(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def invoke(self, question: str) -> RAGResult:
        counter = CallCounter()
        started = time.perf_counter()
        state = self.graph.invoke(
            {"question": question, "query": question, "retrievals": 0, "generations": 0, "trace": []},
            config={"callbacks": [counter], "recursion_limit": 40},
        )
        return RAGResult(
            question=question,
            answer=state.get("generation", ""),
            documents=state.get("documents", []),
            trace=state.get("trace", []),
            llm_calls=counter.calls,
            input_tokens=counter.input_tokens,
            output_tokens=counter.output_tokens,
            seconds=time.perf_counter() - started,
            extra={"action": state.get("action")} if state.get("action") else {},
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
