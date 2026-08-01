"""Baseline RAG: BM25 retrieval + cite-your-sources generation.

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.rag [--limit N]
Reads: runs/chunks.json, questions/questions.json
Writes: runs/answers.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.anthropic_client import call_anthropic
from src.bm25 import BM25Okapi
from src.prompts import build_rag_prompt

CHUNKS_PATH = Path("runs/chunks.json")
QUESTIONS_PATH = Path("questions/questions.json")
OUT_PATH = Path("runs/answers.json")

MAX_TOKENS = 2000
TOP_K = 5

CITATION_RE = re.compile(r"\[(c_\d+)\]")


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text())


def load_questions(limit: int | None) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    return questions[:limit] if limit else questions


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def retrieve(bm25: BM25Okapi, chunks: list[dict], question: str, k: int = TOP_K) -> list[dict]:
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in ranked[:k]]


def generate(question: str, chunks: list[dict]) -> tuple[str, str]:
    system_prompt, user_prompt = build_rag_prompt(question, chunks)
    return call_anthropic(system_prompt, user_prompt, MAX_TOKENS)


def parse_citations(answer_text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(answer_text)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunks = load_chunks()
    bm25 = build_bm25(chunks)
    questions = load_questions(args.limit)

    log: list[str] = []
    answers: list[dict] = []
    for q in questions:
        retrieved = retrieve(bm25, chunks, q["question"])
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        answer_text, stop_reason = generate(q["question"], retrieved)
        cited_ids = parse_citations(answer_text)

        if stop_reason == "max_tokens":
            log.append(f"{q['id']}: TRUNCATED (stop_reason=max_tokens) -- trailing citations may be lost")

        stray = [c for c in cited_ids if c not in retrieved_ids]
        if stray:
            log.append(f"{q['id']}: cited chunk_id(s) not in retrieved set: {stray}")

        answers.append(
            {
                "question_id": q["id"],
                "answer_text": answer_text,
                "retrieved_chunk_ids": retrieved_ids,
                "cited_chunk_ids": cited_ids,
                "stop_reason": stop_reason,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(answers, indent=2, ensure_ascii=False))

    if log:
        print("\n".join(log))
    print(f"\nWrote {len(answers)} answers to {OUT_PATH}")


if __name__ == "__main__":
    main()
