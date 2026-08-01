"""Sample claims across verdict types for hand-labeling (PLAN.md section 7).

The verifier's verdict and reasoning are deliberately never read into or
written out of this script -- the point is an independent human judgment to
compare against the verifier later.

Zero third-party imports. Run with system python3 (not .venv):
  python3 -m src.sample_for_labels
Reads: runs/verdicts.json, runs/chunks.json
Writes: questions/hand_labels_blank.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

VERDICTS_PATH = Path("runs/verdicts.json")
CHUNKS_PATH = Path("runs/chunks.json")
OUT_PATH = Path("questions/hand_labels_blank.json")

SAMPLE_SEED = 42
SUPPORTED_SAMPLE_SIZE = 9


def main() -> None:
    verdicts = json.loads(VERDICTS_PATH.read_text())
    chunk_map = {c["chunk_id"]: c["text"] for c in json.loads(CHUNKS_PATH.read_text())}

    partial = [v for v in verdicts if v["verdict"] == "PARTIAL"]
    unsupported = [v for v in verdicts if v["verdict"] == "UNSUPPORTED"]
    supported = [v for v in verdicts if v["verdict"] == "SUPPORTED"]

    rng = random.Random(SAMPLE_SEED)
    supported_sample = rng.sample(supported, SUPPORTED_SAMPLE_SIZE)

    selected = partial + unsupported + supported_sample
    rng.shuffle(selected)  # break the positional PARTIAL/UNSUPPORTED/SUPPORTED grouping

    out = []
    for v in selected:
        cited_chunks = [{"chunk_id": cid, "text": chunk_map[cid]} for cid in v["cited_chunk_ids"]]
        out.append(
            {
                "claim_id": v["claim_id"],
                "claim_text": v["claim_text"],
                "cited_chunks": cited_chunks,
                "verdict": "",
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    for item in out:
        print(item["claim_id"])
        print(f"  claim: {item['claim_text']}")
        for c in item["cited_chunks"]:
            print(f"  --- {c['chunk_id']} ---")
            print(f"  {c['text']}")
        print()

    print(f"Wrote {len(out)} claims to {OUT_PATH}")


if __name__ == "__main__":
    main()
