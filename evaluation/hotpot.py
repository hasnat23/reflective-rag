"""Compare naive RAG, CRAG and Self-RAG on HotpotQA (distractor setting).

The corpus is the union of the 10 context paragraphs (2 gold + 8 distractors)
of every sampled question. Two conditions:

* ``standard``      the corpus contains the gold paragraphs.
* ``knowledge_gap`` gold paragraphs are removed, so a correct answer requires
                    falling back to external knowledge (Wikipedia for CRAG).

    python evaluation/hotpot.py --n 50
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from datasets import load_dataset

from reflective_rag import PIPELINES, Corpus, Settings
from reflective_rag.llm import get_chat_model, get_embeddings

RESULTS = Path(__file__).parent / "results"


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, gold: str) -> dict:
    return {
        "em": float(normalize(prediction) == normalize(gold)),
        "f1": f1(prediction, gold),
        # Small models often answer in a sentence; "contains" credits a correct answer inside it.
        "contains": float(normalize(gold) in normalize(prediction)),
    }


def load_questions(n: int, seed: int):
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = random.Random(seed).sample(range(len(data)), n)
    return [data[i] for i in idx]


def build_corpus(embeddings, questions, drop_gold: bool) -> Corpus:
    seen, texts, titles = set(), [], []
    gold = {t for q in questions for t in q["supporting_facts"]["title"]} if drop_gold else set()
    for q in questions:
        for title, sentences in zip(q["context"]["title"], q["context"]["sentences"]):
            if title in seen or title in gold:
                continue
            seen.add(title)
            texts.append("".join(sentences).strip())
            titles.append(title)
    return Corpus.from_texts(embeddings, texts, titles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--methods", nargs="+", default=list(PIPELINES))
    parser.add_argument("--conditions", nargs="+", default=["standard", "knowledge_gap"])
    args = parser.parse_args()

    settings = Settings(answer_style="short")
    llm, embeddings = get_chat_model(settings), get_embeddings(settings)
    questions = load_questions(args.n, args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"model": f"{settings.provider}:{settings.model}", "n": args.n, "seed": args.seed, "results": {}}

    for condition in args.conditions:
        corpus = build_corpus(embeddings, questions, drop_gold=condition == "knowledge_gap")
        for method in args.methods:
            pipeline = PIPELINES[method](llm, corpus, settings)
            rows = []
            log = RESULTS / f"{condition}_{method}.jsonl"
            with log.open("w") as f:
                for i, q in enumerate(questions):
                    try:
                        result = pipeline.invoke(q["question"])
                        row = {
                            "id": q["id"],
                            "question": q["question"],
                            "gold": q["answer"],
                            "answer": result.answer,
                            **score(result.answer, q["answer"]),
                            "llm_calls": result.llm_calls,
                            "seconds": result.seconds,
                            "trace": result.trace,
                            "action": result.extra.get("action"),
                        }
                    except Exception as exc:  # noqa: BLE001 - keep going; count as wrong
                        row = {"id": q["id"], "error": repr(exc), "em": 0.0, "f1": 0.0, "contains": 0.0, "llm_calls": 0, "seconds": 0.0}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[{condition}/{method}] {i + 1}/{len(questions)} f1={row['f1']:.2f} {time.strftime('%H:%M:%S')}", flush=True)
            agg = {k: round(mean(r[k] for r in rows), 4) for k in ["em", "f1", "contains", "llm_calls", "seconds"]}
            agg["errors"] = sum("error" in r for r in rows)
            if method == "crag":
                agg["actions"] = dict(Counter(r.get("action") for r in rows))
            summary["results"][f"{condition}/{method}"] = agg
            print(json.dumps({f"{condition}/{method}": agg}), flush=True)
            (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
