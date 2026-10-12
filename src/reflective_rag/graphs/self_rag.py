"""Self-Reflective RAG (Asai et al., 2023, arXiv:2310.11511), as a prompted graph.

Self-RAG trains an LM to emit reflection tokens. This implementation keeps the
same decision points but makes each one a prompted judgement:

- [Retrieve]  is retrieval needed at all?
- [IsRel]     is each retrieved passage relevant?        (filter)
- [IsSup]     is the generation supported by passages?   (regenerate if not)
- [IsUse]     does the generation answer the question?   (rewrite query + retrieve again if not)

Loops are bounded by ``max_retrievals`` and ``max_generations``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .base import Pipeline, RAGState


class SelfRAG(Pipeline):
    name = "self_rag"

    def build(self):
        s = self.settings

        def decide(state: RAGState) -> RAGState:
            needed = self.graders.needs_retrieval(state["question"])
            return {"trace": [f"[Retrieve] {'yes' if needed else 'no'}"], "action": "retrieve" if needed else "direct"}

        def retrieve(state: RAGState) -> RAGState:
            docs = self.corpus.search(state["query"], s.top_k)
            n = state.get("retrievals", 0) + 1
            return {"documents": docs, "retrievals": n, "trace": [f"retrieve#{n}({state['query']!r}): {len(docs)} docs"]}

        def grade(state: RAGState) -> RAGState:
            scores = [self.graders.relevance(state["question"], d) for d in state["documents"]]
            kept = [d for d, x in zip(state["documents"], scores) if x >= s.upper_threshold - 1]
            return {"documents": kept, "trace": [f"[IsRel] scores={scores} kept={len(kept)}"]}

        def rewrite(state: RAGState) -> RAGState:
            query = self.graders.rewrite(state["question"])
            return {"query": query, "trace": [f"rewrite -> {query!r}"]}

        def generate(state: RAGState) -> RAGState:
            n = state.get("generations", 0) + 1
            docs = state.get("documents", []) if state.get("action") != "direct" else []
            return {"generation": self.graders.generate(state["question"], docs), "generations": n, "trace": [f"generate#{n}"]}

        def after_decide(state: RAGState) -> str:
            return "retrieve" if state["action"] == "retrieve" else "generate"

        def after_grade(state: RAGState) -> str:
            if state["documents"] or state["retrievals"] >= s.max_retrievals:
                return "generate"
            return "rewrite"

        def reflect(state: RAGState) -> RAGState:
            grounded = None
            # [IsSup]: only meaningful when there are passages to check against.
            if state.get("documents") and state.get("action") != "direct":
                grounded = self.graders.grounded(state["documents"], state["generation"])
                if not grounded and state["generations"] < s.max_generations:
                    return {"grounded": False, "useful": None, "trace": ["[IsSup] no -> regenerate"]}
            useful = self.graders.useful(state["question"], state["generation"])
            sup = "" if grounded is None else f"[IsSup] {'yes' if grounded else 'no'} "
            return {"grounded": grounded, "useful": useful, "trace": [f"{sup}[IsUse] {'yes' if useful else 'no'}"]}

        def after_reflect(state: RAGState) -> str:
            if state.get("grounded") is False and state.get("useful") is None:
                return "generate"
            if state["useful"] or state.get("action") == "direct" or state["retrievals"] >= s.max_retrievals:
                return "end"
            return "rewrite"

        graph = StateGraph(RAGState)
        for name, fn in [("decide", decide), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite), ("generate", generate), ("reflect", reflect)]:
            graph.add_node(name, fn)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", after_decide, {"retrieve": "retrieve", "generate": "generate"})
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", "reflect")
        graph.add_conditional_edges("reflect", after_reflect, {"generate": "generate", "rewrite": "rewrite", "end": END})
        return graph.compile()
