"""Baseline: retrieve top-k, then generate."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .base import Pipeline, RAGState


class NaiveRAG(Pipeline):
    name = "naive"

    def build(self):
        def retrieve(state: RAGState) -> RAGState:
            docs = self.corpus.search(state["question"], self.settings.top_k)
            return {"documents": docs, "trace": [f"retrieve: {len(docs)} docs"]}

        def generate(state: RAGState) -> RAGState:
            return {"generation": self.graders.generate(state["question"], state["documents"]), "trace": ["generate"]}

        graph = StateGraph(RAGState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()
