"""Compare hand labels to the verifier's verdicts (PLAN.md section 7).

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.agreement
Reads: questions/hand_labels.json, runs/verdicts.json
Writes: nothing (stdout report only)
"""

from __future__ import annotations

import json
from pathlib import Path

HAND_LABELS_PATH = Path("questions/hand_labels.json")
VERDICTS_PATH = Path("runs/verdicts.json")

VERDICT_ORDER = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED")


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (n/a)"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def main() -> None:
    hand_labels = {h["claim_id"]: h["verdict"] for h in load_json(HAND_LABELS_PATH)}
    verifier = {v["claim_id"]: v["verdict"] for v in load_json(VERDICTS_PATH)}

    pairs = [(claim_id, human, verifier[claim_id]) for claim_id, human in hand_labels.items()]

    matrix = {h: {m: 0 for m in VERDICT_ORDER} for h in VERDICT_ORDER}
    for _, human, model in pairs:
        matrix[human][model] += 1

    matches = sum(1 for _, human, model in pairs if human == model)
    n = len(pairs)

    print(f"Percent agreement: {fmt_rate(matches, n)}")
    print()

    print("Confusion matrix (rows = hand label, columns = verifier verdict):")
    print(" " * 14 + "".join(f"{m:>13}" for m in VERDICT_ORDER))
    for h in VERDICT_ORDER:
        print(f"{h:<14}" + "".join(f"{matrix[h][m]:>13}" for m in VERDICT_ORDER))
    print()

    print("Agreement by verifier's verdict:")
    for m in VERDICT_ORDER:
        column_total = sum(matrix[h][m] for h in VERDICT_ORDER)
        print(f"  {m:<12} {fmt_rate(matrix[m][m], column_total)}")
    print()

    print(
        "Note: this 25-claim sample deliberately over-represents non-SUPPORTED\n"
        "verifier verdicts -- all 15 PARTIAL and the 1 UNSUPPORTED claim from the\n"
        "full 86-claim run, plus 9 SUPPORTED chosen at random -- to get enough\n"
        "non-SUPPORTED cases to check. The verifier's overall verdict distribution\n"
        "is ~81% SUPPORTED; this sample is 9/25 (36%) SUPPORTED. The percent-\n"
        "agreement figure above is therefore NOT comparable to what agreement on\n"
        "a random sample of claims would show."
    )


if __name__ == "__main__":
    main()
