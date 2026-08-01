"""Sentence splitting for RAG answer text. Pure stdlib, no API dependency --
kept separate from decompose.py so reporting scripts (src/table.py) can use
it without pulling in src.anthropic_client's API-key check at import time.
"""

from __future__ import annotations

import re

# Sentence splitting: a period/!/? followed by whitespace-or-end is a candidate
# boundary. Digit-glued dots ("$1,500.00", "s.7.4.4", "OAP1-EN.4") never match
# this in the first place since nothing follows the dot but another digit --
# no whitespace. What's left to guard explicitly: known abbreviations, and
# single-letter abbreviations in general (list markers, "s." for section).
SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
NON_CONTENT_RE = re.compile(r"^[\s\-=|]*$")  # blank lines, table separators, horizontal rules
SINGLE_LETTER_ABBR_RE = re.compile(r"(?:^|[\s(])[A-Za-z]\.$")
ABBREVIATIONS = ("e.g.", "i.e.", "a.m.", "p.m.", "no.", "vs.", "etc.", "mr.", "mrs.", "dr.", "jr.", "sr.")


def _is_sentence_boundary(line: str, start: int, end: int) -> bool:
    prefix = line[:end]
    before_char = line[start - 1 : start] if start > 0 else ""
    after_char = line[end : end + 1]
    if before_char.isdigit() and after_char.isdigit():
        return False
    if prefix.lower().endswith(ABBREVIATIONS):
        return False
    if SINGLE_LETTER_ABBR_RE.search(prefix):
        return False
    return True


def _split_line(line: str) -> list[str]:
    parts = []
    start = 0
    for m in SENTENCE_END_RE.finditer(line):
        if _is_sentence_boundary(line, m.start(), m.end()):
            parts.append(line[start : m.end()].strip())
            start = m.end()
    tail = line[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def split_sentences(text: str) -> list[str]:
    sentences = []
    for raw_line in text.splitlines():
        if NON_CONTENT_RE.match(raw_line):
            continue
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[-•*>]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        if not line:
            continue
        sentences.extend(_split_line(line))
    return sentences
