"""Grounded explanation layer. Gemini writes the prose; evidence and verdicts come from retrieval."""
from __future__ import annotations

import json
import re
from typing import List

from app.config import Settings
from app.models import Explanation, SkillFinding, Suggestion

SYSTEM_RULES = """You are a careful resume reviewer. You are given verdicts that were already computed
by a retrieval system, with evidence snippets labelled by id.

Rules:
- Use ONLY the verdicts and evidence below. Never invent skills, employers, metrics or experience.
- Do not mention any technology that does not appear in the findings list.
- For advice about a skill that has evidence, cite the evidence ids you relied on.
- For missing skills, give concrete advice (what kind of project/bullet would prove it) with empty evidence_ids.
- Keep the summary to 2-3 sentences.

Return ONLY JSON: {"summary": str, "suggestions": [{"skill": str, "advice": str, "evidence_ids": [str]}]}
"""


def _render_findings(findings: List[SkillFinding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"- {f.skill} [{f.importance}] -> {f.status} ({f.reason})")
        for e in f.evidence:
            lines.append(f"    [{e.chunk_id}] ({e.section}) {e.text}")
    return "\n".join(lines)


class TemplateExplainer:
    """Deterministic fallback: no API key, no network, still useful."""

    def explain(self, role: str, findings: List[SkillFinding], coverage_score: int) -> Explanation:
        demo = [f.skill for f in findings if f.status == "demonstrated"]
        gaps = [f for f in findings if f.status != "demonstrated"]
        role_txt = f" for the {role} role" if role else ""
        summary = (f"The resume demonstrates {len(demo)} of {len(findings)} skills requested{role_txt} "
                   f"(coverage {coverage_score}%).")
        if demo:
            summary += " Strongest evidence: " + ", ".join(demo[:4]) + "."
        req_gaps = [f.skill for f in gaps if f.importance == "required"]
        if req_gaps:
            summary += " Required skills needing attention: " + ", ".join(req_gaps[:4]) + "."

        gaps.sort(key=lambda f: (f.importance != "required", f.status != "missing"))
        suggestions = []
        for f in gaps[:6]:
            if f.status == "missing":
                advice = (f"{f.skill} is not evidenced anywhere. If you have used it, add a bullet with the task, "
                          f"the tool and a measurable result; otherwise build a small project that uses it.")
            else:
                advice = (f"{f.skill} appears without proof of use. Move it into a project or experience bullet "
                          f"that states what you built and the outcome.")
            suggestions.append(Suggestion(f.skill, advice, [e.chunk_id for e in f.evidence]))
        return Explanation(summary, suggestions, llm_used=False)


class GeminiExplainer:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def explain(self, role: str, findings: List[SkillFinding], coverage_score: int) -> Explanation:
        prompt = (f"{SYSTEM_RULES}\nTarget role: {role or 'not specified'}\nCoverage score: {coverage_score}%\n"
                  f"Findings:\n{_render_findings(findings)}")
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip())
        suggestions = [Suggestion(str(s.get("skill", "")), str(s.get("advice", "")),
                                  [str(i) for i in s.get("evidence_ids", [])])
                       for s in data.get("suggestions", [])]
        return Explanation(str(data.get("summary", "")), suggestions, llm_used=True)


class SafeExplainer:
    """Try the LLM; on any failure fall back to the template so the API never breaks."""

    def __init__(self, primary, fallback: TemplateExplainer):
        self.primary, self.fallback = primary, fallback

    def explain(self, role, findings, coverage_score) -> Explanation:
        if self.primary is not None:
            try:
                return self.primary.explain(role, findings, coverage_score)
            except Exception as e:  # network, quota, bad JSON ...
                out = self.fallback.explain(role, findings, coverage_score)
                out.llm_error = f"{type(e).__name__}: {e}"[:300]
                return out
        return self.fallback.explain(role, findings, coverage_score)


def get_explainer(settings: Settings) -> SafeExplainer:
    primary = None
    if settings.use_llm and settings.gemini_api_key:
        try:
            primary = GeminiExplainer(settings.gemini_api_key, settings.gemini_model)
        except Exception:
            primary = None
    return SafeExplainer(primary, TemplateExplainer())
