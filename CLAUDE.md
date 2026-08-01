# Ledger

Claim-level grounding eval for RAG over insurance policy documents.
Read PLAN.md for the full design. It is the spec. Do not deviate without asking.

## What this measures

Standard RAG evals (RAGAS faithfulness, DeepEval) verify each claim against the
ENTIRE retrieved context. Ledger verifies each claim against ONLY the chunk that
claim cites. A claim can pass the standard test while the citation attached to it
says something different, because another chunk in the top-5 happened to cover it.
That distinction is the point of this project. Do not "improve" it away.

## Corpus

- corpus/oap1.pdf  - OAP1-EN.4 (2026), effective July 1 2026, 61 pages.
  Sections 1-8, subsections numbered (1.4.2, 5.7.1, 7.4.4). Clean text layer.
- corpus/opcf20.pdf - AF-142E, Coverage for Transportation Replacement, 1 page.
- corpus/opcf27.pdf - AF-137E, Liability for Damage to Non-Owned Automobiles, 2 pages.

OPCF 20 explicitly replaces OAP 1 section 7.4.4. That interaction is deliberate
test material. Chunking must not lose it.

parent_section is "ALL" for policy-wide exclusions (s.1.8.1 - s.1.8.5).
Coverage-specific exclusions carry their parent coverage section number.

Excluded from chunking: oap1.pdf front matter (pages ii-vii, including the
"What Insurance is Required by Law" chart, which the document itself
states is superseded by the numbered sections) and the trailing statutory-
conditions cross-reference table.

## Environment

- Python 3.11+. Pipeline stages have zero third-party imports and run with
  the system `python3`: `python3 -m src.<module>`.
- Exception: src/ingest.py imports pdfplumber, so it still needs the venv
  (.venv, deps in requirements.txt): `.venv/bin/python -m src.ingest`.
  chunks.json is already generated -- ingest does not need to be re-run
  unless the chunking logic changes.
- Do not add third-party deps back to rag.py/decompose.py/verify.py/etc.
  On this machine, first-run codesign/Gatekeeper validation of newly
  installed compiled packages (numpy, pydantic-core, ...) inside the venv
  took 10-20+ minutes before macOS cached the verdict. Stdlib-only avoids
  it entirely.
- API key in .env, loaded via a five-line parser in src/rag.py (no
  python-dotenv). NEVER print or log the key.

## Hard rules

- No vector DB, no web UI, no auth, no framework. Plain Python scripts.
- Retrieval is BM25 (src/bm25.py, pure Python, k1=1.5, b=0.75) over chunks.
  In memory. That's it.
- Every pipeline stage reads JSON from runs/ and writes JSON to runs/.
  Stages are independently re-runnable.
- All LLM calls: model claude-sonnet-4-6, temperature 0, via a direct
  HTTPS POST to the Anthropic Messages API using urllib.request (no SDK).
  All prompts live in src/prompts.py, nowhere else.
- LLM calls that expect JSON must strip markdown fences before parsing
  and retry once on parse failure.
- The verifier (src/verify.py) must NEVER receive the original question
  or the full answer. Only one atomic claim + the text of the chunks that
  claim cites. This is the core methodological rule of the project.

## Style

- Small modules, functions under ~40 lines, type hints.
- No classes unless state genuinely requires it.
- After writing any module, run it on real data and show me the output.
  Do not tell me it works. Show me.

## Cost discipline

- Default to running on the first 3 questions (--limit 3) during development.
  Full runs only when I explicitly say so.

## Working agreement

- Plan mode before every phase. I approve the plan before you write code.
- One phase per prompt. Never build multiple phases in one go.
- Commit after every phase.
