"""Split each RAG answer into atomic, self-contained claims.

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.decompose [--limit N]
Reads: runs/answers.json
Writes: runs/claims.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.anthropic_client import call_anthropic
from src.prompts import build_decompose_prompt
from src.sentences import split_sentences

ANSWERS_PATH = Path("runs/answers.json")
OUT_PATH = Path("runs/claims.json")
OUT_UNCITED_PATH = Path("runs/uncited.json")

MAX_TOKENS = 2000

CITATION_RE = re.compile(r"\[(c_\d+)\]")


def load_answers(limit: int | None) -> list[dict]:
    answers = json.loads(ANSWERS_PATH.read_text())
    return answers[:limit] if limit else answers


def sentences_with_citations(answer_text: str) -> list[tuple[str, list[str]]]:
    result = []
    for sent in split_sentences(answer_text):
        cites = CITATION_RE.findall(sent)
        clean = re.sub(r"\s+", " ", CITATION_RE.sub("", sent)).strip()
        result.append((clean, cites))
    return result


def parse_json_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    return json.loads(stripped)


def decompose_answer(question_id: str, answer_text: str) -> tuple[list[dict], int, list[str]]:
    sentence_cites = sentences_with_citations(answer_text)
    sentences = [s for s, _ in sentence_cites]
    cited_indices = [i for i, (_, cites) in enumerate(sentence_cites) if cites]
    uncited_sentences = [s for i, s in enumerate(sentences) if i not in cited_indices]

    if not cited_indices:
        return [], 0, uncited_sentences

    system_prompt, user_prompt = build_decompose_prompt(answer_text, sentences, cited_indices)
    response_text, _ = call_anthropic(system_prompt, user_prompt, MAX_TOKENS)
    try:
        data = parse_json_response(response_text)
    except json.JSONDecodeError:
        response_text, _ = call_anthropic(system_prompt, user_prompt, MAX_TOKENS)
        try:
            data = parse_json_response(response_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{question_id}: could not parse decomposition JSON after retry") from e

    claims = []
    n = 0
    for result in data.get("results", []):
        idx = result.get("sentence_index")
        if idx not in cited_indices:
            continue
        cited_chunk_ids = sentence_cites[idx][1]
        for claim_text in result.get("claims", []):
            n += 1
            claims.append(
                {
                    "question_id": question_id,
                    "claim_id": f"{question_id}_c{n}",
                    "claim_text": claim_text,
                    "cited_chunk_ids": cited_chunk_ids,
                }
            )
    return claims, len(cited_indices), uncited_sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    answers = load_answers(args.limit)

    log: list[str] = []
    all_claims: list[dict] = []
    all_uncited: list[dict] = []
    for a in answers:
        claims, cited_count, uncited_sentences = decompose_answer(a["question_id"], a["answer_text"])
        suffix = ", refusal" if cited_count == 0 else ""
        log.append(f"{a['question_id']}: {cited_count} cited, {len(uncited_sentences)} uncited sentences{suffix}")
        all_claims.extend(claims)
        all_uncited.append(
            {
                "question_id": a["question_id"],
                "uncited_sentence_count": len(uncited_sentences),
                "uncited_sentences": uncited_sentences,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_claims, indent=2, ensure_ascii=False))
    OUT_UNCITED_PATH.write_text(json.dumps(all_uncited, indent=2, ensure_ascii=False))

    print("\n".join(log))
    print(f"\nWrote {len(all_claims)} claims from {len(answers)} answers to {OUT_PATH}")
    print(f"Wrote uncited-sentence log for {len(all_uncited)} answers to {OUT_UNCITED_PATH}")


if __name__ == "__main__":
    main()
