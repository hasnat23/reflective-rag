"""LLM-prompted graders: the "reflection" steps of CRAG and Self-RAG.

The papers train dedicated models for these judgements (a T5 retrieval
evaluator in CRAG, reflection tokens in Self-RAG). Here each one is a prompted
call to the same chat model, which makes the method usable with any LLM at the
cost of some accuracy. Replies are parsed leniently because small local
models do not always return clean JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

RELEVANCE = """You grade whether a retrieved passage helps answer a question.
Question: {question}
Passage: {passage}
Rate relevance from 1 (unrelated) to 5 (contains the answer or key facts needed).
Reply with JSON only: {{"score": <1-5>}}"""

STRIPS = """Below are numbered sentences from a passage. Which ones are useful for answering the question?
Question: {question}
{strips}
Reply with JSON only: {{"useful": [<sentence numbers>]}}. Use [] if none are useful."""

REWRITE = """Rewrite the question as a short keyword search query for Wikipedia.
Question: {question}
Reply with JSON only: {{"query": "<search query>"}}"""

NEEDS_RETRIEVAL = """Does answering this question require looking up specific facts (people, dates, places, events, numbers)?
Question: {question}
Reply with JSON only: {{"retrieve": true}} or {{"retrieve": false}}"""

GROUNDED = """Is every factual claim in the answer supported by the documents?
Documents:
{documents}
Answer: {answer}
Reply with JSON only: {{"grounded": true}} or {{"grounded": false}}"""

USEFUL = """Does the answer actually answer the question (not a refusal, not off-topic)?
Question: {question}
Answer: {answer}
Reply with JSON only: {{"useful": true}} or {{"useful": false}}"""

GENERATE = """Answer the question using only the documents below. If they do not contain the answer, say "I don't know".
Documents:
{documents}
Question: {question}
{style}"""

GENERATE_DIRECT = """Answer the question.
Question: {question}
{style}"""

STYLES = {
    "short": "Reply with the answer only, in as few words as possible (e.g. a name, date, number, or yes/no).",
    "full": "Answer in one to three sentences.",
}


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply; {} if there is none."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


def _ask(llm: BaseChatModel, prompt: str) -> str:
    return str(llm.invoke(prompt).content)


def _bool(reply: str, key: str, default: bool) -> bool:
    value = parse_json(reply).get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes"}
    lowered = reply.lower()
    if re.search(r"\b(true|yes)\b", lowered):
        return True
    if re.search(r"\b(false|no)\b", lowered):
        return False
    return default


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] {d.metadata.get('title', '')}\n{d.page_content}" for i, d in enumerate(documents))


class Graders:
    def __init__(self, llm: BaseChatModel, answer_style: str = "full") -> None:
        self.llm = llm
        self.style = STYLES[answer_style]

    def relevance(self, question: str, document: Document) -> int:
        reply = _ask(self.llm, RELEVANCE.format(question=question, passage=document.page_content[:2000]))
        score = parse_json(reply).get("score")
        if score is None:
            digits = re.findall(r"[1-5]", reply)
            score = digits[0] if digits else 3
        try:
            return min(max(int(float(score)), 1), 5)
        except (TypeError, ValueError):
            return 3

    def useful_strips(self, question: str, strips: list[str]) -> list[int]:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(strips))
        reply = _ask(self.llm, STRIPS.format(question=question, strips=numbered))
        picked = parse_json(reply).get("useful")
        if not isinstance(picked, list):
            return list(range(len(strips)))  # unparsable: keep everything
        return sorted({int(i) - 1 for i in picked if str(i).isdigit() and 0 < int(i) <= len(strips)})

    def rewrite(self, question: str) -> str:
        query = parse_json(_ask(self.llm, REWRITE.format(question=question))).get("query")
        return query.strip() if isinstance(query, str) and query.strip() else question

    def needs_retrieval(self, question: str) -> bool:
        return _bool(_ask(self.llm, NEEDS_RETRIEVAL.format(question=question)), "retrieve", default=True)

    def grounded(self, documents: list[Document], answer: str) -> bool:
        prompt = GROUNDED.format(documents=format_documents(documents), answer=answer)
        return _bool(_ask(self.llm, prompt), "grounded", default=True)

    def useful(self, question: str, answer: str) -> bool:
        return _bool(_ask(self.llm, USEFUL.format(question=question, answer=answer)), "useful", default=True)

    def generate(self, question: str, documents: list[Document]) -> str:
        if not documents:
            return _ask(self.llm, GENERATE_DIRECT.format(question=question, style=self.style)).strip()
        prompt = GENERATE.format(documents=format_documents(documents), question=question, style=self.style)
        return _ask(self.llm, prompt).strip()
