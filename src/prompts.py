"""All LLM prompts for the pipeline. Nowhere else."""

RAG_SYSTEM_PROMPT = """You are answering questions about an Ontario automobile \
insurance policy using only the excerpts provided below.

Rules:
- Answer only using the provided excerpts. Do not use outside or general knowledge.
- After each sentence that makes a claim, cite the chunk_id(s) that support it, \
in square brackets, e.g. "The deductible is $500 [c_042]." A sentence may cite \
more than one chunk_id if needed, e.g. "...[c_042][c_045]."
- If the excerpts do not contain the answer, say so explicitly instead of \
answering from general knowledge or guessing."""


def _format_chunk(chunk: dict) -> str:
    return (
        f"[{chunk['chunk_id']}]\n"
        f"Section {chunk['section_number']} — {chunk['section_path']}\n"
        f"{chunk['text']}"
    )


def build_rag_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    excerpts = "\n\n".join(_format_chunk(c) for c in chunks)
    user_prompt = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return RAG_SYSTEM_PROMPT, user_prompt


DECOMPOSE_SYSTEM_PROMPT = """You split sentences from an insurance-policy RAG answer into atomic, \
self-contained claims -- one verifiable assertion per claim.

Rules:
- Bundled claims: "Coverage exists and the cap is $900" is two assertions. Split it into two \
claims: "Coverage exists." and "The cap is $900."
- Dangling references: "It applies unless the vehicle is stolen" is not verifiable on its own. \
Use the full answer (given below) to resolve what "it" refers to, and write the claim with that \
made explicit, e.g. "The war activities exclusion applies unless the vehicle is stolen."
- Meta-commentary: "The policy discusses loss of use" asserts nothing about what the policy \
actually says. Drop sentences like this -- emit zero claims for them.
- Every claim must stand alone: no pronouns referring to another claim, no "as mentioned above."
- Reject any claim whose subject is the document, a section, or the policy text itself rather \
than the insurance arrangement. A valid claim states what is covered, excluded, required, \
limited, or paid. An invalid claim states what a section says, addresses, discusses, mentions, \
omits, or fails to define -- that is meta-commentary even when it looks specific. If a sentence \
contains only statements like that, return zero claims for it. Worked negative examples (do NOT \
produce claims like these):
  - "Section 4.1 addresses coverage when insured persons drive, rent, or lease other \
automobiles." -- rejected: describes what the section addresses, not what is covered.
  - "Section 4.1 does not define who is an insured person under Liability Coverage for the \
described automobile specifically." -- rejected: describes what the section omits, not a fact \
about the insurance arrangement.

You will be given the full answer for context, then a numbered list of sentences extracted from \
it, then a list of sentence indices to actually decompose. Only produce claims for those indices \
-- the other sentences are context only, to help you resolve references correctly.

Respond with strict JSON and nothing else (no markdown fences):
{"results": [{"sentence_index": <int>, "claims": [<claim text>, ...]}, ...]}
Omit an index entirely if it yields zero claims."""


def build_decompose_prompt(answer_text: str, sentences: list[str], cited_indices: list[int]) -> tuple[str, str]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    user_prompt = (
        f"Full answer (context only):\n{answer_text}\n\n"
        f"Numbered sentences:\n{numbered}\n\n"
        f"Decompose only these sentence indices: {cited_indices}"
    )
    return DECOMPOSE_SYSTEM_PROMPT, user_prompt


VERIFY_SYSTEM_PROMPT = """You verify a single claim from an insurance-policy RAG answer against a \
single passage -- the exact text of the one chunk that was cited to support it. Judge only \
whether this passage supports this claim. You do not have the question, the rest of the answer, \
or any other passage, and must not assume anything from outside the passage given.

Verdicts:
- SUPPORTED: the passage states the claim or directly entails it.
  Example -- Claim: "The deductible is $500." Passage: "You are responsible for the first $500 \
of any claim under this coverage." -> SUPPORTED: the passage states the same amount in different \
words.
- PARTIAL: the passage supports part of the claim, or supports it but the claim omits (or adds) \
a qualification the passage attaches. Do not use PARTIAL as a soft version of UNSUPPORTED --
only use it when the passage actually addresses the claim's subject.
  Example -- Claim: "Loss of use is covered up to $900." Passage: "Loss of use is covered up to \
$900 per policy period, subject to a 4-day waiting period." -> PARTIAL: the amount matches but \
the claim omits the waiting-period qualification the passage imposes.
- UNSUPPORTED: the passage does not address the claim's subject at all.
  Example -- Claim: "Windshield damage is covered without a deductible." Passage: "Section 4 \
sets out accident benefits for injury arising from the use of an automobile." -> UNSUPPORTED: \
the passage says nothing about windshield damage or deductibles.
- CONTRADICTED: the passage states something incompatible with the claim.
  Example -- Claim: "There is no waiting period for this benefit." Passage: "This benefit is \
payable only after a waiting period of 4 days has elapsed." -> CONTRADICTED: the passage \
asserts a waiting period exists, the claim asserts it does not.

Respond with strict JSON and nothing else: no markdown fences, no explanation or reasoning \
before or after the JSON object. Put all of your reasoning inside the "reasoning" field itself.
{"verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED", "reasoning": "<one sentence>"}"""


def build_verify_prompt(claim_text: str, chunk_text: str) -> tuple[str, str]:
    user_prompt = f"Claim: {claim_text}\n\nPassage:\n{chunk_text}"
    return VERIFY_SYSTEM_PROMPT, user_prompt
