"""Blind claim verification.

Each claim is checked against ONLY the verbatim text of the chunk(s) it
cites -- never the question, the rest of the answer, or any other chunk.
For claims citing more than one chunk, each cited chunk is checked
separately and the verdicts are combined; chunks are never concatenated,
since that would let the verifier assemble support by stitching together
halves of two different passages.

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.verify [--limit N]
Reads: runs/claims.json, runs/chunks.json
Writes: runs/verdicts.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.anthropic_client import call_anthropic
from src.prompts import build_verify_prompt

CLAIMS_PATH = Path("runs/claims.json")
CHUNKS_PATH = Path("runs/chunks.json")
OUT_PATH = Path("runs/verdicts.json")

MAX_TOKENS = 300

VERDICTS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED")
STRENGTH = {"CONTRADICTED": 3, "PARTIAL": 2, "UNSUPPORTED": 1}


def load_chunk_map() -> dict[str, str]:
    chunks = json.loads(CHUNKS_PATH.read_text())
    return {c["chunk_id"]: c["text"] for c in chunks}


def load_claims(limit: int | None) -> list[dict]:
    claims = json.loads(CLAIMS_PATH.read_text())
    if limit is None:
        return claims
    seen_question_ids: list[str] = []
    for c in claims:
        if c["question_id"] not in seen_question_ids:
            seen_question_ids.append(c["question_id"])
    allowed = set(seen_question_ids[:limit])
    return [c for c in claims if c["question_id"] in allowed]


def parse_json_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    return json.loads(stripped)


def verify_against_chunk(claim_text: str, chunk_text: str, chunk_id: str) -> dict:
    system_prompt, user_prompt = build_verify_prompt(claim_text, chunk_text)

    for attempt in range(2):
        response_text, _ = call_anthropic(system_prompt, user_prompt, MAX_TOKENS)
        try:
            data = parse_json_response(response_text)
            verdict = data["verdict"]
            reasoning = data["reasoning"]
            if verdict not in VERDICTS:
                raise ValueError(f"unknown verdict {verdict!r}")
            return {"chunk_id": chunk_id, "verdict": verdict, "reasoning": reasoning}
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 1:
                raise RuntimeError(f"could not parse verify JSON for chunk {chunk_id} after retry")
    raise AssertionError("unreachable")


def combine_verdicts(per_chunk: list[dict]) -> tuple[str, str]:
    for result in per_chunk:
        if result["verdict"] == "SUPPORTED":
            return "SUPPORTED", result["reasoning"]
    strongest = max(per_chunk, key=lambda r: STRENGTH[r["verdict"]])
    return strongest["verdict"], strongest["reasoning"]


def verify_claim(claim_text: str, cited_chunk_ids: list[str], chunk_map: dict[str, str]) -> dict:
    per_chunk = [
        verify_against_chunk(claim_text, chunk_map[chunk_id], chunk_id) for chunk_id in cited_chunk_ids
    ]
    verdict, reasoning = combine_verdicts(per_chunk)
    return {"verdict": verdict, "reasoning": reasoning, "per_chunk_verdicts": per_chunk}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunk_map = load_chunk_map()
    claims = load_claims(args.limit)

    log: list[str] = []
    verdicts: list[dict] = []
    for claim in claims:
        result = verify_claim(claim["claim_text"], claim["cited_chunk_ids"], chunk_map)
        verdicts.append(
            {
                "claim_id": claim["claim_id"],
                "question_id": claim["question_id"],
                "claim_text": claim["claim_text"],
                "cited_chunk_ids": claim["cited_chunk_ids"],
                "verdict": result["verdict"],
                "reasoning": result["reasoning"],
                "per_chunk_verdicts": result["per_chunk_verdicts"],
            }
        )
        log.append(f"{claim['claim_id']}: {result['verdict']} -- {result['reasoning']}")
        if len(result["per_chunk_verdicts"]) > 1:
            for pc in result["per_chunk_verdicts"]:
                log.append(f"    {pc['chunk_id']}: {pc['verdict']} -- {pc['reasoning']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False))

    print("\n".join(log))
    print(f"\nWrote {len(verdicts)} verdicts to {OUT_PATH}")


if __name__ == "__main__":
    main()
