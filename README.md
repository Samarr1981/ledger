# Ledger

A claim-level grounding eval for RAG over insurance policy documents.

TL;DR: An ordinary RAG pipeline over Ontario's standard auto policy refused 4 of 13 answerable questions, every one a retrieval miss with the answering clause sitting in the corpus. No citation metric detects this, because a refusal carries no citations to check. Of the claims the system did make, 81.4% were grounded in their cited passage with zero fabrications. The verifier agreed with my hand labels 52% of the time, so read that grounding number as a floor.

Most RAG systems report citation coverage: the percentage of answer sentences carrying a source reference. That number proves a chunk was retrieved and attached. It proves nothing about whether the sentence is true relative to that chunk.

Ledger splits every answer into atomic claims and verifies each one against **only the chunk that claim cites**, in a call that never sees the question or the rest of the answer.

## Results

Corpus: the OAP 1 (Ontario Automobile Policy, the standard contract every private auto policy in Ontario is written on) plus the OPCF 20 and OPCF 27 endorsements. 104 chunks.

Baseline: BM25 top-5, cite-your-sources prompt, `claude-sonnet-4-6` at temperature 0. Deliberately ordinary.

16 hand-written questions. 86 atomic claims.

| Metric | Result |
|---|---|
| Claim-level grounding | 70/86 (81.4%) |
| Citation coverage, all questions | 43/85 (50.6%) |
| Citation coverage, answered questions only | 32/52 (61.5%) |
| Refusals | 7/16 |
| Correct refusals (trap questions) | 3/3 |
| **False refusals on answerable questions** | **4/13 (30.8%)** |
| **Verifier agreement with hand labels** | **13/25 (52%)** |

### The finding that survives

Four of thirteen non-trap questions were refused with the answer sitting in the corpus. Every one was a BM25 retrieval miss:

| Question | Missed chunk |
|---|---|
| Who counts as an insured person under Liability Coverage | `c_030` — OAP 1 s.3.2 Who is Covered |
| What happens if I refuse an inspection | `c_027` — OAP 1 s.2.5 Inspection |
| Will the policy pay for a rental while my car is repaired | `c_098` — OPCF 20, the whole endorsement |
| Does my policy cover damage to a car I rented | `c_018` — OAP 1 s.2.2.2 Temporary Substitute Automobile |

No citation metric detects this failure, because a refusal carries no citations to check.

Meanwhile all three questions deliberately written to be unanswerable from the corpus were correctly refused. The system handled the traps and failed the ordinary cases.

### The finding about the eval itself

I hand-labeled 25 claims against their cited passages and compared to the verifier. Agreement was 52%.

The disagreement is systematic rather than random: 7 of the 15 claims the verifier called PARTIAL were hand-labeled SUPPORTED. The verifier hedges downward, which makes 81.4% a floor rather than a ceiling. More importantly, two careful readers applying the same four-verdict rubric to the same passages agreed barely half the time. The SUPPORTED/PARTIAL boundary is not reproducible enough to carry a headline number.

The sample deliberately over-represents disputed cases (all 15 PARTIAL, the 1 UNSUPPORTED, plus 9 SUPPORTED sampled at random), so this figure is not comparable to agreement on a random sample.

### Other patterns

**Dropped qualifications.** 15 PARTIAL verdicts, 6 from one chunk. The passage says the insurer pays the difference between deductibles *when the owner's deductible is larger and the damage exceeds it*. The answer says the insurer pays the difference.

**Inference cited as source.** Three claims assert that Comprehensive covers what Collision does not. The passage lists what each covers and never states the exclusion. The model inferred it from the coverages being mutually exclusive, then cited the passage as though it had said so.

**Zero fabrications. Zero CONTRADICTED verdicts.** This system's failure mode is not inventing things.

## How it differs from existing work

Claim decomposition plus per-claim verification is the standard definition of RAG faithfulness. RAGAS, DeepEval, and RAGChecker all do it. ALCE is the established benchmark for citation quality.

Ledger differs on one axis: RAGAS-style faithfulness verifies each claim against the **entire retrieved context**, so a claim passes if anything in the top-k supports it, even when the citation attached to it says something else. Ledger verifies against **only the cited chunks**, which is the test that matches what a reader experiences when they click a citation and read the clause.

For multi-chunk claims, each cited chunk is verified separately rather than concatenated. Concatenation lets the verifier assemble support from halves of two passages. One claim in this run (a combined `$350` deductible figure) exists only by summing across two chunks and is correctly scored PARTIAL as a result.

## Pipeline

Each stage reads JSON from `runs/` and writes JSON to `runs/`, so any stage can be re-run without repeating the API calls of the ones before it.

| Stage | Module | Output |
|---|---|---|
| Ingest | `src/ingest.py` | `chunks.json` — verbatim, structure-aware chunks |
| Retrieve + generate | `src/rag.py` | `answers.json` |
| Decompose | `src/decompose.py` | `claims.json`, `uncited.json` |
| Verify | `src/verify.py` | `verdicts.json` |
| Report | `src/table.py` | stdout |
| Agreement | `src/agreement.py` | stdout |

Chunk text is verbatim. Exclusions carry a `parent_section` reference to the coverage they modify, since an exclusion severed from its coverage produces confidently wrong answers. Policy-wide exclusions carry `"ALL"`.

One page (the coverage-by-vehicle-type grid on p.10) required table extraction and is flagged `reconstructed: true` in the data, because its layout is reassembled rather than verbatim.

## Running it

Zero third-party dependencies at runtime. Stock Python 3.11+, standard library only, including a pure-Python BM25 and direct `urllib` calls to the API.

```bash
# Corpus (not committed, Crown copyright)
# Download from fsrao.ca into corpus/:
#   oap1.pdf    OAP 1, OAP1-EN.4 (2026)
#   opcf20.pdf  OPCF 20, AF-142E
#   opcf27.pdf  OPCF 27, AF-137E

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Ingest needs pdfplumber; everything else does not
pip install pdfplumber
python3 -m src.ingest

python3 -m src.rag --limit 3
python3 -m src.decompose --limit 3
python3 -m src.verify --limit 3
python3 -m src.table
python3 -m src.agreement
```

Drop `--limit` for a full run. Output from the run reported above is committed under `results/`.

## Limitations

- **Small sample.** 16 questions, 86 claims. Directional, not conclusive.
- **The verifier disagrees with a human half the time.** Quantified above rather than assumed away. The grounding rate should be read as a floor.
- **Decomposition is a dependency.** An early version emitted meta-commentary claims ("Section 4.1 addresses coverage when...") that would have scored SUPPORTED trivially and inflated the grounding rate. Caught by reading claims by hand, not by any aggregate.
- **Single corpus, single domain.** Nothing here shows the gap generalizes beyond Ontario auto policy documents.
- **Single generator, single retriever.** One model at one temperature, lexical retrieval only.
- **Ledger measures whether a claim is supported by its citation, not whether the answer addresses the question.** One question in this run was answered with correct citations to a provision that does not apply to it. The harness scores those claims as supported and does not catch the error.

## Next

Swap BM25 for embedding retrieval, change nothing else, re-run the identical harness. Same questions, same verifier, one variable. Production systems typically run both (hybrid retrieval), and the four false refusals here are all lexical-mismatch failures the dense half would plausibly catch.

Also deferred from this version: failure attribution (retrieval / generation / fabrication), a confidence gate, and an HTML report.

## Corpus source

OAP 1 and the OPCF endorsements are published by the Financial Services Regulatory Authority of Ontario at [fsrao.ca](https://www.fsrao.ca). © King's Printer for Ontario. Not redistributed here.
