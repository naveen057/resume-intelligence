"""Skill matching: retrieval finds evidence, deterministic rules decide the verdict.

The LLM never decides whether a skill is demonstrated. That keeps verdicts reproducible,
testable, and immune to hallucination. The LLM only explains verdicts afterwards.
"""
from __future__ import annotations

from typing import Dict, List

from app.config import Settings
from app.ingestion.chunker import chunk_text
from app.models import Chunk, Evidence, Hit, SkillFinding
from app.analysis.skills import JDSkill, SkillExtractor
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever

WEIGHTS = {"required": 1.0, "preferred": 0.5}
WEAK_SECTIONS = {"skills", "summary", "education"}   # claims, not proof of use
STATUS_VALUE = {"demonstrated": 1.0, "partial": 0.5, "missing": 0.0}


def _evidence(hit: Hit, match_type: str) -> Evidence:
    c = hit.chunk
    return Evidence(c.id, c.text, c.section, c.context, round(hit.score, 4),
                    None if hit.rerank_score is None else round(hit.rerank_score, 4), match_type)


class ResumeMatcher:
    def __init__(self, extractor: SkillExtractor, embedder: Embedder, reranker, settings: Settings):
        self.extractor, self.embedder, self.reranker, self.settings = extractor, embedder, reranker, settings

    def chunk_resume(self, resume_text: str) -> List[Chunk]:
        return chunk_text(resume_text, source="resume")

    def match(self, chunks: List[Chunk], jd_skills: List[JDSkill]) -> List[SkillFinding]:
        retriever = HybridRetriever(chunks, self.embedder)
        findings = [self._match_skill(retriever, chunks, s) for s in jd_skills]
        return self._apply_alternatives(findings, jd_skills)

    @staticmethod
    def _apply_alternatives(findings: List[SkillFinding], jd_skills: List[JDSkill]) -> List[SkillFinding]:
        """'FastAPI or Flask': if one alternative is demonstrated, don't report the other as a gap."""
        by = {f.skill: f for f in findings}
        drop = set()
        for jd in jd_skills:
            f = by[jd.name]
            satisfied_by = next((a for a in jd.alternatives if a in by and by[a].status == "demonstrated"), None)
            if f.status != "demonstrated" and satisfied_by:
                drop.add(jd.name)
                note = f" Also satisfies the JD's 'or' alternative: {jd.name}."
                if note not in by[satisfied_by].reason:
                    by[satisfied_by].reason += note
        return [f for f in findings if f.skill not in drop]

    def _match_skill(self, retriever: HybridRetriever, chunks: List[Chunk], jd: JDSkill) -> SkillFinding:
        cfg = self.settings
        aliases = self.extractor.skills[jd.name]["aliases"]
        lex_query = " ".join([jd.name, *aliases])
        dense_query = f"{jd.name}: {jd.context}"
        ranked = retriever.search(lex_query, k=len(chunks), mode="hybrid", dense_query=dense_query)

        # 1) Keyword evidence: scan *every* chunk so a mention is never missed due to top-k cutoffs.
        keyword = [h for h in ranked if self.extractor.mentions(jd.name, h.chunk.text)]
        # 2) Candidate pool for reranking = keyword hits + top semantic candidates.
        pool: Dict[str, Hit] = {h.chunk.id: h for h in keyword[:4] + ranked[: cfg.candidate_k]}
        reranked = self.reranker.rerank(f"Experience with {jd.name}", list(pool.values()))
        score_of = {h.chunk.id: h for h in reranked}
        keyword = [score_of[h.chunk.id] for h in keyword]
        keyword.sort(key=lambda h: (h.rerank_score or 0.0, h.score), reverse=True)

        strong = [h for h in keyword if h.chunk.section not in WEAK_SECTIONS]
        weak = [h for h in keyword if h.chunk.section in WEAK_SECTIONS]
        kw_ids = {h.chunk.id for h in keyword}
        semantic = [h for h in reranked if h.chunk.id not in kw_ids and h.rerank_score is not None
                    and h.rerank_score >= cfg.semantic_threshold][: cfg.evidence_k]

        if strong:
            status, reason = "demonstrated", "Used in experience/project evidence."
            conf = 0.75 + (0.1 if len(strong) >= 2 else 0) + (0.1 if (strong[0].rerank_score or 0) >= 0.5 else 0)
            ev = [_evidence(h, "keyword") for h in strong[: cfg.evidence_k]]
            if weak and len(ev) < cfg.evidence_k:
                ev.append(_evidence(weak[0], "keyword"))
        elif weak:
            status, conf = "partial", 0.4
            reason = (f"Only mentioned in the {weak[0].chunk.section} section; "
                      "no experience or project bullet shows it in use.")
            ev = [_evidence(weak[0], "keyword")]
        elif semantic:
            status, reason = "partial", "Related work found, but the skill is never named explicitly."
            conf = round(0.3 + 0.2 * (semantic[0].rerank_score or 0), 2)
            ev = [_evidence(h, "semantic") for h in semantic]
        else:
            status, conf, ev, reason = "missing", 0.8, [], "No mention or related evidence found in the resume."

        return SkillFinding(jd.name, jd.category, jd.importance, status, round(min(conf, 1.0), 2), reason, ev)


def coverage(findings: List[SkillFinding]) -> int:
    total = sum(WEIGHTS[f.importance] for f in findings)
    if not total:
        return 0
    got = sum(WEIGHTS[f.importance] * STATUS_VALUE[f.status] for f in findings)
    return round(100 * got / total)
