"""Rule-based faithfulness checks on the final output (no LLM judge needed).

1. Every 'demonstrated' verdict is backed by evidence that really contains the skill.
2. Every evidence id cited in a suggestion exists in the retrieved evidence.
3. Every skill named in the summary/advice appears in the JD or resume (no hallucinated tech).
"""
from __future__ import annotations

from typing import List

from app.analysis.skills import SkillExtractor
from app.models import Explanation, FaithfulnessReport, SkillFinding


def check_faithfulness(findings: List[SkillFinding], explanation: Explanation, extractor: SkillExtractor,
                       resume_text: str, jd_text: str) -> FaithfulnessReport:
    issues: List[str] = []
    checks = 0

    for f in findings:
        if f.status == "demonstrated":
            checks += 1
            if not any(extractor.mentions(f.skill, e.text) for e in f.evidence):
                issues.append(f"'{f.skill}' marked demonstrated without supporting evidence text")

    valid_ids = {e.chunk_id for f in findings for e in f.evidence}
    for s in explanation.suggestions:
        for cid in s.evidence_ids:
            checks += 1
            if cid not in valid_ids:
                issues.append(f"Suggestion for '{s.skill}' cites unknown evidence id '{cid}'")

    allowed = extractor.find(resume_text) | extractor.find(jd_text)
    narrative = " ".join([explanation.summary] + [s.advice for s in explanation.suggestions])
    for skill in extractor.find(narrative):
        checks += 1
        if skill not in allowed:
            issues.append(f"Narrative mentions '{skill}', which is in neither the resume nor the JD")

    passed = checks - len(issues)
    return FaithfulnessReport(score=round(passed / checks, 3) if checks else 1.0,
                              checks=checks, passed=passed, issues=issues)
