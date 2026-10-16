"""Structure-aware chunking: one bullet (or short paragraph) = one chunk.

Resume bullets are already self-contained claims, so bullet-level chunks give precise
citations. Section names are kept as metadata and used later to tell "listed in Skills"
apart from "used in a project".
"""
from __future__ import annotations

import re
from typing import List

from app.models import Chunk

_SECTION_ALIASES = {
    "summary": "summary", "professional summary": "summary", "objective": "summary", "profile": "summary",
    "experience": "experience", "work experience": "experience", "employment": "experience",
    "projects": "projects", "project": "projects",
    "skills": "skills", "technical skills": "skills",
    "education": "education",
    "certifications": "certifications", "certification": "certifications",
    "achievements": "achievements", "awards": "achievements", "publications": "publications",
}
_SECTION_RE = re.compile(
    r"^\s*(" + "|".join(sorted(map(re.escape, _SECTION_ALIASES), key=len, reverse=True)) + r")\s*:?\s*$", re.I
)
_BULLET_RE = re.compile(r"^\s*([-•*–▪●◦]|\d+[.)])\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, max_words: int) -> List[str]:
    if len(text.split()) <= max_words:
        return [text]
    parts, cur = [], []
    for sent in _SENTENCE_RE.split(text):
        if cur and len(" ".join(cur + [sent]).split()) > max_words:
            parts.append(" ".join(cur))
            cur = []
        cur.append(sent)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_text(text: str, source: str = "resume", max_words: int = 80) -> List[Chunk]:
    chunks: List[Chunk] = []
    section, context = "general", ""
    buf: List[str] = []

    def flush() -> None:
        if not buf:
            return
        for piece in _split_long(" ".join(buf).strip(), max_words):
            if len(piece.split()) >= 2:
                chunks.append(Chunk(id=f"{source}-{len(chunks)}", text=piece, section=section,
                                    source=source, context=context))
        buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        m = _SECTION_RE.match(line)
        if m:
            flush()
            section, context = _SECTION_ALIASES[m.group(1).lower()], ""
            continue
        if _BULLET_RE.match(line):
            flush()
            buf.append(_BULLET_RE.sub("", line))
        elif section == "skills":
            flush()
            buf.append(line)       # every skills line stands alone ("Languages: Python, SQL")
            flush()
        elif (section in {"experience", "projects"} and not buf
              and len(line.split()) <= 14 and not line.endswith(".")):
            context = line         # job / project title line: metadata, not a chunk
        else:
            buf.append(line)
    flush()
    return chunks
