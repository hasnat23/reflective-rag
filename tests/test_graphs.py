"""Graph routing tests with a scripted LLM and deterministic embeddings (no network, no models)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from reflective_rag import Corpus, CorrectiveRAG, NaiveRAG, SelfRAG, Settings
from reflective_rag.graders import Graders, parse_json
from reflective_rag.graphs.crag import split_strips

TEXTS = [
    # Three sentences each -> two strips -> exactly one refine call per kept doc.
    "Paris is the capital of France. It has 2 million residents. It lies on the Seine.",
    "The Eiffel Tower was completed in 1889. It is in Paris. It is 330 m tall.",
    "Bananas are rich in potassium. They are yellow. They grow in the tropics.",
]


class FakeWeb:
    def __init__(self):
        self.queries = []

    def search(self, query, k=3):
        self.queries.append(query)
        return [Document(page_content="Web fact about the question.", metadata={"title": "Web", "source": "wikipedia"})]


@pytest.fixture
def corpus():
    return Corpus.from_texts(DeterministicFakeEmbedding(size=32), TEXTS, ["paris", "eiffel", "banana"])


def settings(**kw):
    return Settings(top_k=2, **kw)


def llm(*responses):
    return FakeListChatModel(responses=list(responses))


class TestParsing:
    def test_parse_json_with_noise(self):
        assert parse_json('Sure! {"score": 4} hope that helps') == {"score": 4}
        assert parse_json("no json here") == {}

    def test_relevance_fallbacks(self, corpus):
        g = Graders(llm("I'd say 5 out of 5", "{}"))
        doc = Document(page_content="x")
        assert g.relevance("q", doc) == 5
        assert g.relevance("q", doc) == 3  # unparsable -> neutral

    def test_bool_fallbacks(self):
        g = Graders(llm('{"grounded": "yes"}', "No, it is not.", "???"))
        docs = [Document(page_content="x")]
        assert g.grounded(docs, "a") is True
        assert g.grounded(docs, "a") is False
        assert g.grounded(docs, "a") is True  # default

    def test_strips(self):
        assert split_strips("A one. B two. C three.", 2) == ["A one. B two.", "C three."]
        g = Graders(llm('{"useful": [2, 9, "x"]}'))
        assert g.useful_strips("q", ["a", "b", "c"]) == [1]


def test_naive(corpus):
    result = NaiveRAG(llm("Paris"), corpus, settings(), web=FakeWeb()).invoke("Capital of France?")
    assert result.answer == "Paris" and len(result.documents) == 2 and result.llm_calls == 1


class TestCRAG:
    def test_correct_path_refines_without_web(self, corpus):
        web = FakeWeb()
        # relevance x2, one strip-filter call per kept doc, generate
        model = llm('{"score": 5}', '{"score": 3}', '{"useful": [1]}', '{"useful": [1]}', "Paris")
        result = CorrectiveRAG(model, corpus, settings(), web=web).invoke("Capital of France?")
        assert result.extra["action"] == "correct"
        assert web.queries == []
        assert result.answer == "Paris"

    def test_incorrect_path_uses_web_only(self, corpus):
        web = FakeWeb()
        model = llm('{"score": 1}', '{"score": 2}', '{"query": "france capital"}', "Answer from web")
        result = CorrectiveRAG(model, corpus, settings(), web=web).invoke("Capital of France?")
        assert result.extra["action"] == "incorrect"
        assert web.queries == ["france capital"]
        assert [d.metadata["source"] for d in result.documents] == ["wikipedia"]

    def test_ambiguous_path_combines(self, corpus):
        web = FakeWeb()
        model = llm('{"score": 3}', '{"score": 1}', '{"query": "q"}', '{"useful": [1, 2]}', "Combined")
        result = CorrectiveRAG(model, corpus, settings(), web=web).invoke("Capital of France?")
        assert result.extra["action"] == "ambiguous"
        assert {d.metadata["source"] for d in result.documents} == {"corpus", "wikipedia"}
        assert result.answer == "Combined"


class TestSelfRAG:
    def test_happy_path(self, corpus):
        model = llm('{"retrieve": true}', '{"score": 5}', '{"score": 4}', "Paris", '{"grounded": true}', '{"useful": true}')
        result = SelfRAG(model, corpus, settings(), web=FakeWeb()).invoke("Capital of France?")
        assert result.answer == "Paris"
        assert result.trace[-1] == "[IsSup] yes [IsUse] yes"

    def test_no_retrieval(self, corpus):
        model = llm('{"retrieve": false}', "Hello!", '{"useful": true}')
        result = SelfRAG(model, corpus, settings(), web=FakeWeb()).invoke("Say hello")
        assert result.trace[0] == "[Retrieve] no" and result.answer == "Hello!"

    def test_regenerates_when_unsupported(self, corpus):
        model = llm(
            '{"retrieve": true}', '{"score": 5}', '{"score": 5}',
            "Lyon", '{"grounded": false}',
            "Paris", '{"grounded": true}', '{"useful": true}',
        )  # fmt: skip
        result = SelfRAG(model, corpus, settings(max_generations=2), web=FakeWeb()).invoke("Capital of France?")
        assert result.answer == "Paris" and "[IsSup] no -> regenerate" in result.trace

    def test_rewrites_when_nothing_relevant(self, corpus):
        model = llm(
            '{"retrieve": true}', '{"score": 1}', '{"score": 1}',
            '{"query": "france capital city"}', '{"score": 5}', '{"score": 1}',
            "Paris", '{"grounded": true}', '{"useful": true}',
        )  # fmt: skip
        result = SelfRAG(model, corpus, settings(max_retrievals=2), web=FakeWeb()).invoke("Capital of France?")
        assert any(t.startswith("retrieve#2('france capital city')") for t in result.trace)
        assert result.answer == "Paris"

    def test_loops_are_bounded(self, corpus):
        # Every grader says "no"; the graph must still terminate.
        model = llm('{"retrieve": true}', '{"score": 1}', '{"score": 1}', '{"query": "q"}', '{"score": 1}', '{"score": 1}', "x", '{"useful": false}')
        result = SelfRAG(model, corpus, settings(max_retrievals=2), web=FakeWeb()).invoke("?")
        assert result.answer == "x"


def test_mermaid_renders(corpus):
    assert "web_search" in CorrectiveRAG(llm("{}"), corpus, settings()).mermaid()
