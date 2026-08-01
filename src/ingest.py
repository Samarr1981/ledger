"""Parse the corpus PDFs into structure-aware, verbatim chunks.

Run: .venv/bin/python -m src.ingest
Writes: runs/chunks.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pdfplumber

CORPUS_DIR = Path("corpus")
OUT_PATH = Path("runs/chunks.json")
MAX_WORDS = 380
BARE_HEADER_MAX_WORDS = 20  # shortest genuine leaf in the corpus is 32 words

TOP_SECTION_RE = re.compile(r"^Section\s+(\d+)$")
NUMERAL_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?$")
CONDITION_NUM_RE = re.compile(r"^(\d{1,2})\.\s")
ROMAN_FOOTER_RE = re.compile(r"^[ivx]+$", re.IGNORECASE)
OAP1_FOOTER_RE = re.compile(r"^OAP1-EN\.4 \(2026\) Page \d+$")
COPYRIGHT_RE = re.compile(r"^©")
PAGE_OF_RE = re.compile(r"^Page \d+ of \d+$")
BULLET_RE = re.compile(r"^([•\-]|\([a-zA-Z0-9.]+\)|\d{1,2}\.)\s")


@dataclass
class Word:
    text: str
    top: int
    size: float
    bold: bool


@dataclass
class Line:
    words: list[Word]
    page: int

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def all_bold(self) -> bool:
        return all(w.bold for w in self.words)

    @property
    def first_bold(self) -> bool:
        return self.words[0].bold


@dataclass
class Leaf:
    section_number: str
    end_number: str
    title: str
    page: int
    path_prefix: str
    lines: list[tuple[int, str]] = field(default_factory=list)  # (page, text)
    reconstructed: bool = False
    no_merge: bool = False

    def word_count(self) -> int:
        return sum(len(t.split()) for _, t in self.lines)


LINE_TOLERANCE = 3  # px; words within this vertical band belong to one visual line


def extract_lines(page, page_num: int) -> list[Line]:
    words = sorted(page.extract_words(extra_attrs=["size", "fontname"]), key=lambda w: (w["top"], w["x0"]))
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["top"] - groups[-1][0]["top"] <= LINE_TOLERANCE:
            groups[-1].append(w)
        else:
            groups.append([w])
    lines = []
    for group in groups:
        top = round(group[0]["top"])
        ordered = sorted(group, key=lambda w: w["x0"])
        ws = [Word(w["text"], top, w["size"], "Bold" in w["fontname"]) for w in ordered]
        lines.append(Line(ws, page_num))
    return lines


def strip_footer(lines: list[Line]) -> list[Line]:
    kept = lines[:]
    while kept:
        t = kept[-1].text
        if (
            OAP1_FOOTER_RE.match(t)
            or COPYRIGHT_RE.match(t)
            or ROMAN_FOOTER_RE.match(t)
            or PAGE_OF_RE.match(t)
            or "Effective (" in t
            or t.startswith("(AF-")
        ):
            kept.pop()
        else:
            break
    return kept


def join_lines(lines: list[tuple[int, str]]) -> str:
    out = ""
    for _, text in lines:
        if not out:
            out = text
        elif BULLET_RE.match(text):
            out += "\n" + text
        else:
            out += " " + text
    return out


def dot_drop(number: str) -> str | None:
    parts = number.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def top_of(number: str) -> str:
    return number.split(".")[0]


def strip_number_prefix(text: str, number: str) -> str:
    rest = text[len(number):]
    return rest.lstrip(". ").strip() if text.startswith(number) else text.strip()


def group_key(leaf: Leaf, oap1: bool) -> tuple:
    # OAP1: only the finest-grain level (N.N.N, e.g. 7.2.1/7.2.2/7.2.3) merges
    # by shared parent. Coarser levels (N.N named subsections like 7.1/7.2, or
    # Section 8's N.N condition-number leaves) merge only with an exact same
    # number, so distinct topics never combine, but Section 8's repeated
    # same-condition sub-blocks (all tagged e.g. "8.2") still do.
    # OPCF has no such wrapper: its top-level clauses (1, 2, 3...) are true
    # siblings meant to merge freely, so plain shared-parent grouping applies.
    depth = leaf.section_number.count(".") + 1
    if not oap1:
        return (None, dot_drop(leaf.section_number), depth)
    parent = dot_drop(leaf.section_number) if depth == 3 else leaf.section_number
    return (top_of(leaf.section_number), parent, depth)


def merge_leaves(leaves: list[Leaf], oap1: bool) -> list[Leaf]:
    merged: list[Leaf] = []
    current: Leaf | None = None
    for leaf in leaves:
        if current is None:
            current = leaf
            continue
        same_group = group_key(current, oap1) == group_key(leaf, oap1)
        mergeable = same_group and not current.no_merge and not leaf.no_merge
        if mergeable and current.word_count() + leaf.word_count() <= MAX_WORDS:
            current = replace(current, end_number=leaf.section_number, lines=current.lines + leaf.lines)
        else:
            merged.append(current)
            current = leaf
    if current is not None:
        merged.append(current)
    return merged


def split_leaf(leaf: Leaf) -> list[Leaf]:
    if leaf.word_count() <= MAX_WORDS:
        return [leaf]
    pieces: list[list[tuple[int, str]]] = [[]]
    for page_num, text in leaf.lines:
        if pieces[-1] and BULLET_RE.match(text):
            words_so_far = sum(len(t.split()) for _, t in pieces[-1])
            if words_so_far >= MAX_WORDS:
                pieces.append([])
        pieces[-1].append((page_num, text))
    return [replace(leaf, lines=p, page=p[0][0]) for p in pieces if p]


def leaf_label(leaf: Leaf) -> str:
    if leaf.section_number == leaf.end_number:
        return f"{leaf.section_number} {leaf.title}".strip()
    return f"{leaf.section_number}-{leaf.end_number}"


def leaf_to_chunk(leaf: Leaf, doc: str, chunk_id: str) -> dict:
    section_number = leaf.section_number
    parent = dot_drop(section_number)
    if re.match(r"^1\.8\.[1-5]$", section_number):
        parent = "ALL"
    path = f"{leaf.path_prefix} > {leaf_label(leaf)}" if leaf.path_prefix else leaf_label(leaf)
    return {
        "chunk_id": chunk_id,
        "doc": doc,
        "section_path": path,
        "section_number": section_number,
        "page": leaf.page,
        "parent_section": parent,
        "text": join_lines(leaf.lines),
        "reconstructed": leaf.reconstructed,
    }


def subsection_number(line: Line) -> str | None:
    first = line.words[0]
    m = NUMERAL_RE.match(first.text)
    if not m:
        return None
    number = m.group(1)
    # Real N.N.N subsection headers are occasionally printed in regular
    # weight (e.g. 4.1.1/4.1.2), unlike everywhere else in the corpus where
    # headers are bold. Bare 1-2 component numerals in regular weight are
    # never real headers here -- they're numbered list items in body prose
    # (or, in Section 8, statutory-condition markers handled separately) --
    # so only relax the bold requirement at full N.N.N depth.
    if first.bold or number.count(".") == 2:
        return number
    return None


def top_section_number(line: Line) -> str | None:
    m = TOP_SECTION_RE.match(line.text)
    return m.group(1) if m else None


OAP1_SKIP_RANGES = [
    (0, 0, "cover page, no content"),
    (2, 4, "table of contents, navigation only"),
    (5, 6, "'What Insurance is Required by Law' chart: defers to the numbered "
           "sections and pdfplumber column extraction is unreliable for it"),
    (66, 67, "trailing statutory-conditions cross-reference table, navigation only"),
]
OAP1_FRONT_MATTER = 1  # "About This Policy"
OAP1_TABLE_PAGE = 16  # index of the page-10 coverage grid
TABLE_TITLE = "What Types of Coverage Extend to Other Automobiles?"
TABLE_COLUMNS = ["Liability", "Accident Benefits", "Uninsured Automobile", "Direct Compensation", "Loss or Damage"]


def oap1_page_skipped(index: int) -> bool:
    return any(lo <= index <= hi for lo, hi, _ in OAP1_SKIP_RANGES)


def skip_log_lines() -> list[str]:
    lines = [f"oap1.pdf: skipped pdf pages {lo + 1}-{hi + 1} ({why})" for lo, hi, why in OAP1_SKIP_RANGES]
    lines.append(
        f"oap1.pdf: pdf page {OAP1_TABLE_PAGE + 1} used table-aware extraction "
        "(extract_tables) for the coverage grid; chunk marked reconstructed=true"
    )
    return lines


def is_bare_header_leaf(leaf: Leaf) -> bool:
    # A header (e.g. "1.6 Our Rights and Responsibilities") that got its own
    # leaf but is immediately followed by a deeper header, with no body text
    # of its own before that happens. These carry no verifiable content and
    # would otherwise become a retrievable, citable chunk that supports
    # nothing. See flush sites in parse_oap1_body / parse_opcf for handling.
    return leaf.word_count() <= BARE_HEADER_MAX_WORDS


def flush(leaves: list[Leaf], current: Leaf | None) -> None:
    if current is not None and current.lines:
        leaves.append(current)


def parse_oap1_body(pages: list) -> list[Leaf]:
    leaves: list[Leaf] = []
    current = Leaf("front_matter", "front_matter", "About This Policy", OAP1_FRONT_MATTER + 1, "")
    section_title = ""
    sub_title = ""
    seen_statutory_marker = False
    for index, page in enumerate(pages):
        if oap1_page_skipped(index):
            continue
        page_num = index + 1
        lines = strip_footer(extract_lines(page, page_num))
        for line in lines:
            if index == OAP1_TABLE_PAGE and line.text.strip() == TABLE_TITLE:
                break  # rest of this page is the table; rebuilt separately below
            top_num = top_section_number(line)
            if top_num:
                flush(leaves, current)
                section_title = line.text
                sub_title = ""
                seen_statutory_marker = False
                current = Leaf(top_num, top_num, "", page_num, section_title)
                continue
            in_section_8 = section_title == "Section 8"
            sub_num = subsection_number(line)
            if sub_num and not in_section_8:
                if current is not None and is_bare_header_leaf(current):
                    if "." not in current.section_number:
                        tagline = " ".join(t for _, t in current.lines)
                        section_title = f"{section_title} {tagline}".strip()
                    current = None
                else:
                    flush(leaves, current)
                depth = sub_num.count(".") + 1
                if depth == 2:
                    sub_title = f"{sub_num} {line.text.split(' ', 1)[1]}" if " " in line.text else sub_num
                prefix = section_title if depth <= 2 else f"{section_title} > {sub_title}"
                current = Leaf(sub_num, sub_num, strip_number_prefix(line.text, sub_num), page_num, prefix, [(page_num, line.text)])
                continue
            if in_section_8 and line.all_bold and not sub_num:
                if not seen_statutory_marker:
                    current.lines.append((page_num, line.text))
                    seen_statutory_marker = line.text.strip() == "Statutory Conditions"
                    continue
                flush(leaves, current)
                current = Leaf("8.0", "8.0", line.text, page_num, section_title, [(page_num, line.text)])
                continue
            current.lines.append((page_num, line.text))
        if index == OAP1_TABLE_PAGE:
            flush(leaves, current)
            table_leaf = build_page10_table_leaf(page, page_num, section_title)
            leaves.append(table_leaf)
            current = Leaf(sub_title.split(" ", 1)[0], sub_title.split(" ", 1)[0], "", page_num, section_title)
    flush(leaves, current)
    return finalize_section8(leaves)


def _cell(row: list, idx: int) -> str:
    val = row[idx] if idx < len(row) else None
    return val.replace("\n", " ").strip() if val else ""


def build_page10_table_leaf(page, page_num: int, section_title: str) -> Leaf:
    raw = page.extract_tables()[0]
    groups: list[list[list]] = []
    for row in raw[4:]:  # rows 0-3 are the (already-known) column headers
        has_data = any(_cell(row, i) for i in range(2, 7))
        if has_data:
            groups.append([row])
        elif groups:
            groups[-1].append(row)  # wrapped row-label continuation, no data of its own
    text_lines = [TABLE_TITLE]
    for group in groups:
        label = " ".join(_cell(r, 1) for r in group if _cell(r, 1))
        if not label:
            continue
        cells = []
        for col_idx, col_name in zip(range(2, 7), TABLE_COLUMNS):
            value = " ".join(_cell(r, col_idx) for r in group if _cell(r, col_idx)) or "(blank)"
            cells.append(f"{col_name}: {value}")
        text_lines.append(f"{label} — " + "; ".join(cells))
    lines = [(page_num, t) for t in text_lines]
    return Leaf("2.2", "2.2", TABLE_TITLE, page_num, section_title, lines, reconstructed=True, no_merge=True)


def finalize_section8(leaves: list[Leaf]) -> list[Leaf]:
    condition_num = 0
    out = []
    for leaf in leaves:
        if leaf.section_number != "8.0":
            out.append(leaf)
            continue
        body_text = leaf.lines[1][1] if len(leaf.lines) > 1 else ""
        m = CONDITION_NUM_RE.match(body_text)
        if m:
            condition_num = int(m.group(1))
        number = f"8.{condition_num}"
        out.append(replace(leaf, section_number=number, end_number=number))
    return out


def parse_opcf(pages: list) -> list[Leaf]:
    leaves: list[Leaf] = []
    current: Leaf | None = None
    title_stack: dict[int, str] = {}
    for index, page in enumerate(pages):
        page_num = index + 1
        lines = strip_footer(extract_lines(page, page_num))
        for line in lines:
            sub_num = subsection_number(line)
            if sub_num:
                depth = sub_num.count(".") + 1
                title_stack[depth] = f"{sub_num} {line.text.split(' ', 1)[1]}" if " " in line.text else sub_num
                prefix = " > ".join(title_stack[d] for d in range(1, depth))
                if current is not None and is_bare_header_leaf(current):
                    current = None
                else:
                    flush(leaves, current)
                current = Leaf(sub_num, sub_num, strip_number_prefix(line.text, sub_num), page_num, prefix, [(page_num, line.text)])
                continue
            if current is not None:
                current.lines.append((page_num, line.text))
    flush(leaves, current)
    return leaves


def build_doc(path: Path, oap1: bool) -> list[Leaf]:
    with pdfplumber.open(path) as pdf:
        pages = list(pdf.pages)
        leaves = parse_oap1_body(pages) if oap1 else parse_opcf(pages)
    split: list[Leaf] = []
    for leaf in merge_leaves(leaves, oap1):
        split.extend(split_leaf(leaf))
    return split


def main() -> None:
    log: list[str] = []
    chunk_id = 0
    chunks: list[dict] = []
    docs = [
        ("oap1", CORPUS_DIR / "oap1.pdf", True),
        ("opcf20", CORPUS_DIR / "opcf20.pdf", False),
        ("opcf27", CORPUS_DIR / "opcf27.pdf", False),
    ]
    for doc, path, oap1 in docs:
        leaves = build_doc(path, oap1)
        for leaf in leaves:
            chunk_id += 1
            chunks.append(leaf_to_chunk(leaf, doc, f"c_{chunk_id:03d}"))
        log.append(f"{doc}: {len(leaves)} chunks")
    log.extend(skip_log_lines())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    print("\n".join(log))
    print(f"\nWrote {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
