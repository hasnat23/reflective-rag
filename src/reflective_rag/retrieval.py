"""Vector-store retrieval over a local corpus, plus Wikipedia as the "web" fallback."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


class Corpus:
    def __init__(self, embeddings: Embeddings) -> None:
        self.store = InMemoryVectorStore(embeddings)
        self.size = 0

    @classmethod
    def from_texts(cls, embeddings: Embeddings, texts: Iterable[str], titles: Iterable[str] | None = None) -> Corpus:
        corpus = cls(embeddings)
        texts = list(texts)
        titles = list(titles) if titles is not None else [""] * len(texts)
        corpus.add([Document(page_content=t, metadata={"title": ti, "source": "corpus"}) for t, ti in zip(texts, titles)])
        return corpus

    def add(self, documents: list[Document]) -> None:
        if documents:
            self.store.add_documents(documents)
            self.size += len(documents)

    def search(self, query: str, k: int) -> list[Document]:
        return self.store.similarity_search(query, k=k) if self.size else []


class WikipediaSearch:
    """Keyless web search over Wikipedia (MediaWiki API); returns article intros."""

    API = "https://en.wikipedia.org/w/api.php"
    USER_AGENT = "reflective-rag/0.1 (https://github.com/hasnat23/reflective-rag)"

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def search(self, query: str, k: int = 3) -> list[Document]:
        titles = self._titles(query, k)
        if not titles and len(query.split()) > 2:
            # Wikipedia requires every term to match; retry with any-term matching.
            titles = self._titles(" OR ".join(query.split()), k)
        if not titles:
            return []
        pages = self._get(
            {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
            }
        )
        by_title = {p.get("title"): p.get("extract", "") for p in pages.get("query", {}).get("pages", {}).values()}
        return [
            Document(page_content=by_title[t], metadata={"title": t, "source": "wikipedia"})
            for t in titles
            if by_title.get(t)
        ]

    def _titles(self, query: str, k: int) -> list[str]:
        hits = self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json"})
        return [h["title"] for h in hits.get("query", {}).get("search", [])]

    def _get(self, params: dict) -> dict:
        url = f"{self.API}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except OSError:
            return {}
