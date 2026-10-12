"""Corrective Retrieval-Augmented Generation (Yan et al., 2024, arXiv:2401.15884).

1. Retrieve from the local corpus.
2. A retrieval evaluator scores each document's relevance and picks an action:
   - Correct:   at least one document scores >= upper threshold
                -> refine the internal documents.
   - Incorrect: every document scores <= lower threshold
                -> discard them, rewrite the query, search the web.
   - Ambiguous: otherwise -> refine internal documents AND search the web.
3. Knowledge refinement ("decompose-then-recompose"): split documents into
   sentence strips, keep only the strips the evaluator marks as useful.
4. Generate from the refined knowledge.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .base import Pipeline, RAGState


def split_strips(text: str, sentences_per_strip: int = 2) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [" ".join(sentences[i : i + sentences_per_strip]) for i in range(0, len(sentences), sentences_per_strip)]


class CorrectiveRAG(Pipeline):
    name = "crag"

    def build(self):
        s = self.settings

        def retrieve(state: RAGState) -> RAGState:
            docs = self.corpus.search(state["question"], s.top_k)
            return {"documents": docs, "trace": [f"retrieve: {len(docs)} docs"]}

        def evaluate(state: RAGState) -> RAGState:
            scores = [self.graders.relevance(state["question"], d) for d in state["documents"]]
            if any(x >= s.upper_threshold for x in scores):
                action = "correct"
            elif all(x <= s.lower_threshold for x in scores):
                action = "incorrect"
            else:
                action = "ambiguous"
            # Keep documents above the lower threshold for refinement.
            kept = [d for d, x in zip(state["documents"], scores) if x > s.lower_threshold]
            return {"documents": kept, "action": action, "trace": [f"evaluate: scores={scores} -> {action}"]}

        def refine(state: RAGState) -> RAGState:
            refined = []
            for doc in state["documents"]:
                strips = split_strips(doc.page_content)
                if len(strips) <= 1:
                    refined.append(doc)
                    continue
                keep = self.graders.useful_strips(state["question"], strips)
                if keep:
                    text = " ".join(strips[i] for i in keep)
                    refined.append(Document(page_content=text, metadata={**doc.metadata, "refined": True}))
            return {"documents": refined, "trace": [f"refine: {len(state['documents'])} -> {len(refined)} docs"]}

        def web_search(state: RAGState) -> RAGState:
            query = self.graders.rewrite(state["question"])
            found = self.web.search(query, s.web_results)
            base = state["documents"] if state.get("action") == "ambiguous" else []
            return {"documents": base + found, "query": query, "trace": [f"web_search({query!r}): {len(found)} docs"]}

        def generate(state: RAGState) -> RAGState:
            return {"generation": self.graders.generate(state["question"], state["documents"]), "trace": ["generate"]}

        def route(state: RAGState) -> str:
            return "refine" if state["action"] == "correct" else "web_search"

        graph = StateGraph(RAGState)
        for name, fn in [("retrieve", retrieve), ("evaluate", evaluate), ("refine", refine), ("web_search", web_search), ("generate", generate)]:
            graph.add_node(name, fn)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges("evaluate", route, {"refine": "refine", "web_search": "web_search"})
        graph.add_edge("web_search", "refine")
        graph.add_edge("refine", "generate")
        graph.add_edge("generate", END)
        return graph.compile()
