"""Orchestration: parse -> chunk -> extract JD skills -> retrieve evidence -> explain -> verify."""
from __future__ import annotations

from typing import Optional

from app.analysis.llm import SafeExplainer, get_explainer
from app.analysis.matcher import ResumeMatcher, coverage
from app.analysis.skills import SkillExtractor
from app.config import Settings
from app.evaluation.faithfulness import check_faithfulness
from app.models import AnalysisResult
from app.retrieval.embedder import get_embedder
from app.retrieval.reranker import get_reranker


class AnalysisPipeline:
    def __init__(self, settings: Settings, extractor: SkillExtractor, matcher: ResumeMatcher,
                 explainer: SafeExplainer):
        self.settings, self.extractor, self.matcher, self.explainer = settings, extractor, matcher, explainer

    def analyze(self, resume_text: str, jd_text: str, target_role: str = "") -> AnalysisResult:
        chunks = self.matcher.chunk_resume(resume_text)
        jd_skills = self.extractor.extract_jd(jd_text)
        findings = self.matcher.match(chunks, jd_skills)
        score = coverage(findings)
        explanation = self.explainer.explain(target_role, findings, score)
        report = check_faithfulness(findings, explanation, self.extractor, resume_text, jd_text)
        return AnalysisResult(
            target_role=target_role, coverage_score=score,
            demonstrated_count=sum(f.status == "demonstrated" for f in findings),
            total_skills=len(findings), findings=findings, summary=explanation.summary,
            suggestions=explanation.suggestions, faithfulness=report,
            llm_used=explanation.llm_used, llm_error=explanation.llm_error,
        )


def build_pipeline(settings: Optional[Settings] = None) -> AnalysisPipeline:
    settings = settings or Settings()
    extractor = SkillExtractor.from_file(settings.taxonomy_path)
    matcher = ResumeMatcher(extractor, get_embedder(settings), get_reranker(settings), settings)
    return AnalysisPipeline(settings, extractor, matcher, get_explainer(settings))
