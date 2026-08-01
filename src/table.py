"""Aggregate report: citation coverage, grounding rate, the gap, verdict
distribution, and refusal accuracy -- per question and overall.

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.table
Reads: runs/answers.json, runs/uncited.json, runs/verdicts.json,
       questions/questions.json
Writes: nothing (stdout report only)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.sentences import split_sentences

ANSWERS_PATH = Path("runs/answers.json")
UNCITED_PATH = Path("runs/uncited.json")
VERDICTS_PATH = Path("runs/verdicts.json")
QUESTIONS_PATH = Path("questions/questions.json")
CHUNKS_PATH = Path("runs/chunks.json")

VERDICT_ORDER = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED")


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (n/a)"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def citation_coverage(answer_text: str, uncited_count: int) -> tuple[int, int]:
    total = len(split_sentences(answer_text))
    return total - uncited_count, total


def verdict_counts(verdicts: list[dict]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICT_ORDER}
    for v in verdicts:
        counts[v["verdict"]] += 1
    return counts


def gap_str(cited: int, total: int, supported: int, claim_total: int) -> str:
    if not total or not claim_total:
        return "n/a"
    pp = 100 * cited / total - 100 * supported / claim_total
    return f"{pp:+.1f} pp"


def main() -> None:
    answers = {a["question_id"]: a["answer_text"] for a in load_json(ANSWERS_PATH)}
    uncited = {u["question_id"]: u["uncited_sentence_count"] for u in load_json(UNCITED_PATH)}
    questions = load_json(QUESTIONS_PATH)

    verdicts_by_q: dict[str, list[dict]] = {}
    for v in load_json(VERDICTS_PATH):
        verdicts_by_q.setdefault(v["question_id"], []).append(v)

    total_cited = total_sentences = 0
    non_refused_cited = non_refused_sentences = 0
    total_supported = total_claims = 0
    overall_counts = {v: 0 for v in VERDICT_ORDER}
    refusals_by_type: dict[str, list[bool]] = {}
    refused_questions: list[dict] = []

    for q in questions:
        qid, qtype = q["id"], q["type"]
        cited, total = citation_coverage(answers.get(qid, ""), uncited.get(qid, 0))
        q_verdicts = verdicts_by_q.get(qid, [])
        counts = verdict_counts(q_verdicts)
        supported = counts["SUPPORTED"]
        claim_total = len(q_verdicts)

        total_cited += cited
        total_sentences += total
        if not q["refused"]:
            non_refused_cited += cited
            non_refused_sentences += total
        total_supported += supported
        total_claims += claim_total
        for k in VERDICT_ORDER:
            overall_counts[k] += counts[k]
        refusals_by_type.setdefault(qtype, []).append(q["refused"])
        if q["refused"]:
            refused_questions.append(q)

        print(f"{qid} [{qtype}]")
        print(f"  citation coverage: {fmt_rate(cited, total)}")
        print(f"  grounding rate:    {fmt_rate(supported, claim_total)}")
        print(f"  gap:               {gap_str(cited, total, supported, claim_total)}")
        print(
            "  verdicts:          "
            + ", ".join(f"{k} {counts[k]}" for k in VERDICT_ORDER)
            + f" (of {claim_total} claims)"
        )
        print(f"  refusal:           {'yes' if q['refused'] else 'no'}")
        print()

    print(f"=== OVERALL ({len(questions)} questions) ===")
    print(f"citation coverage (all {len(questions)} questions):        {fmt_rate(total_cited, total_sentences)}")
    non_refused_n = len(questions) - len(refused_questions)
    print(
        f"citation coverage ({non_refused_n} non-refused questions): "
        f"{fmt_rate(non_refused_cited, non_refused_sentences)}"
    )
    print(f"grounding rate:    {fmt_rate(total_supported, total_claims)}")
    print(f"gap (all {len(questions)} questions):        {gap_str(total_cited, total_sentences, total_supported, total_claims)}")
    print(
        f"gap ({non_refused_n} non-refused questions): "
        f"{gap_str(non_refused_cited, non_refused_sentences, total_supported, total_claims)}"
    )
    print(
        "verdict distribution: "
        + ", ".join(f"{k} {overall_counts[k]}" for k in VERDICT_ORDER)
        + f" (of {total_claims} claims)"
    )
    print()

    print("refusal counts by question type:")
    for qtype, flags in sorted(refusals_by_type.items()):
        print(f"  {qtype:10s} {fmt_rate(sum(flags), len(flags))}")
    print()

    correct = [q for q in refused_questions if q["refusal_correct"]]
    incorrect = [q for q in refused_questions if not q["refusal_correct"]]
    non_trap = [q for q in questions if q["type"] != "trap"]
    non_trap_incorrect = [q for q in non_trap if q["refused"] and not q["refusal_correct"]]

    print(f"refusals: {fmt_rate(len(refused_questions), len(questions))}")
    print(f"  correct:   {fmt_rate(len(correct), len(refused_questions))}")
    print(f"  incorrect: {fmt_rate(len(incorrect), len(refused_questions))}")
    print(f"false refusal rate over non-trap questions: {fmt_rate(len(non_trap_incorrect), len(non_trap))}")
    print()

    chunk_info = {c["chunk_id"]: c for c in load_json(CHUNKS_PATH)}
    print("incorrect refusals (corpus had the answer, retrieval missed it):")
    for q in incorrect:
        missed_id = q["missed_chunk_id"]
        if missed_id:
            c = chunk_info[missed_id]
            location = f"{missed_id} ({c['doc']}, {c['section_path']})"
        else:
            location = "unlabeled"
        print(f"  {q['id']} [{q['type']}] -- missed {location} -- {q['question']}")


if __name__ == "__main__":
    main()
