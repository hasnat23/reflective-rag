"""Ask questions over a folder of text/markdown files.

    rrag ask "Who founded the company?" --docs ./notes --method crag
    rrag graph crag            # print the Mermaid diagram of a pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .graphs import PIPELINES
from .llm import get_chat_model, get_embeddings
from .retrieval import Corpus


def load_folder(folder: Path, chunk_chars: int = 1200) -> tuple[list[str], list[str]]:
    texts, titles = [], []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        paragraphs = [p.strip() for p in path.read_text(errors="ignore").split("\n\n") if p.strip()]
        chunk = ""
        for p in paragraphs:
            if chunk and len(chunk) + len(p) > chunk_chars:
                texts.append(chunk)
                titles.append(path.name)
                chunk = ""
            chunk = f"{chunk}\n\n{p}" if chunk else p
        if chunk:
            texts.append(chunk)
            titles.append(path.name)
    return texts, titles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rrag", description="Corrective RAG and Self-RAG with LangGraph.")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--docs", type=Path, required=True, help="folder of .txt/.md files")
    ask.add_argument("--method", choices=sorted(PIPELINES), default="crag")
    ask.add_argument("--no-web", action="store_true", help="disable the Wikipedia fallback")
    graph = sub.add_parser("graph")
    graph.add_argument("method", choices=sorted(PIPELINES))
    args = parser.parse_args(argv)

    settings = Settings()
    if args.command == "graph":
        # Diagrams do not need real models.
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        print(PIPELINES[args.method](FakeListChatModel(responses=["{}"]), Corpus.__new__(Corpus), settings).mermaid())
        return 0

    texts, titles = load_folder(args.docs)
    corpus = Corpus.from_texts(get_embeddings(settings), texts, titles)
    web = _NoWeb() if args.no_web else None
    pipeline = PIPELINES[args.method](get_chat_model(settings), corpus, settings, web=web)
    result = pipeline.invoke(args.question)
    for step in result.trace:
        print(f"  · {step}")
    print(f"\n{result.answer}\n")
    sources = sorted({d.metadata.get("title", "") for d in result.documents})
    print(f"sources: {', '.join(sources) or '-'}   ({result.llm_calls} LLM calls, {result.seconds:.1f}s)")
    return 0


class _NoWeb:
    def search(self, query: str, k: int = 3):
        return []


if __name__ == "__main__":
    raise SystemExit(main())
