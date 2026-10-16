"""Taxonomy-based skill extraction with alias matching and required/preferred detection."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

_NOT_WORD = r"A-Za-z0-9+#&"


@dataclass
class JDSkill:
    name: str
    category: str
    importance: str       # "required" | "preferred"
    context: str          # the JD line the skill was found in
    alternatives: List[str] = field(default_factory=list)   # 'FastAPI or Flask': either one satisfies


_PREFERRED_HEAD = re.compile(r"nice to have|preferred|bonus|good to have|desirable|a plus|plus points", re.I)
_REQUIRED_HEAD = re.compile(r"requirements?|qualifications|must[- ]have|required|responsibilities|what you|skills", re.I)
_BULLET = re.compile(r"^\s*([-•*–▪●◦]|\d+[.)])\s+")


class SkillExtractor:
    def __init__(self, taxonomy: List[dict]):
        self.skills: Dict[str, dict] = {s["name"]: s for s in taxonomy}
        self._patterns: Dict[str, re.Pattern] = {}
        for s in taxonomy:
            alts = "|".join(re.escape(a) for a in sorted(s["aliases"], key=len, reverse=True))
            flags = 0 if s.get("case_sensitive") else re.I
            self._patterns[s["name"]] = re.compile(rf"(?<![{_NOT_WORD}])(?:{alts})(?![{_NOT_WORD}])", flags)

    @classmethod
    def from_file(cls, path: Path) -> "SkillExtractor":
        return cls(json.loads(Path(path).read_text()))

    def mentions(self, skill: str, text: str) -> bool:
        return bool(self._patterns[skill].search(text))

    def find(self, text: str) -> Set[str]:
        return {name for name in self.skills if self.mentions(name, text)}

    def _alternative_groups(self, line: str, names: List[str]) -> Dict[str, Set[str]]:
        """Skills directly joined by 'or' or '/' ("FastAPI or Flask", "Tableau / Power BI") are
        interchangeable. 'REST APIs with FastAPI or Flask' makes only FastAPI/Flask alternatives."""
        spans = sorted((self._patterns[n].search(line).span(), n) for n in names)
        groups: List[List[str]] = []
        has_or: List[bool] = []
        prev_end = None
        for (start, end), name in spans:
            gap = line[prev_end:start] if prev_end is not None and start >= prev_end else None
            if gap is not None and gap.strip() and re.fullmatch(r"[\s,/]*(?:or)?[\s,/]*", gap, re.I):
                groups[-1].append(name)
                has_or[-1] |= bool(re.search(r"\bor\b|/", gap, re.I))
            else:
                groups.append([name])
                has_or.append(False)
            prev_end = max(end, prev_end or 0)
        return {n: set(g) - {n} for g, ok in zip(groups, has_or) if ok and len(g) > 1 for n in g}

    def extract_jd(self, jd_text: str) -> List[JDSkill]:
        mode = "required"
        found: Dict[str, JDSkill] = {}
        for raw in jd_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            is_heading = not _BULLET.match(line) and len(line.split()) <= 8
            if is_heading and _PREFERRED_HEAD.search(line):
                mode = "preferred"
                continue
            if is_heading and _REQUIRED_HEAD.search(line):
                mode = "required"
                continue
            line_mode = "preferred" if _PREFERRED_HEAD.search(line) else mode
            in_line = [n for n in self.skills if self.mentions(n, line)]
            alt_of = self._alternative_groups(line, in_line)
            for name in in_line:
                prev = found.get(name)
                if prev is None:
                    found[name] = JDSkill(name, self.skills[name]["category"], line_mode, _BULLET.sub("", line))
                elif prev.importance == "preferred" and line_mode == "required":
                    prev.importance = "required"
                if name in alt_of:
                    found[name].alternatives = sorted(set(found[name].alternatives) | alt_of[name])
        return list(found.values())
