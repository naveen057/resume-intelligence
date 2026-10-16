"""Plain dataclasses for the core pipeline. Pydantic is used only at the API boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    id: str
    text: str
    section: str = "general"
    source: str = "resume"
    context: str = ""          # e.g. "Data Science Intern | RetailCo" the bullet belongs to


@dataclass
class Hit:
    chunk: Chunk
    score: float                           # retrieval score (BM25, cosine or RRF)
    rerank_score: Optional[float] = None   # cross-encoder probability in [0, 1], if reranked


@dataclass
class Evidence:
    chunk_id: str
    text: str
    section: str
    context: str
    retrieval_score: float
    rerank_score: Optional[float]
    match_type: str            # "keyword" or "semantic"


@dataclass
class SkillFinding:
    skill: str
    category: str
    importance: str            # "required" | "preferred"
    status: str                # "demonstrated" | "partial" | "missing"
    confidence: float
    reason: str
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Suggestion:
    skill: str
    advice: str
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    summary: str
    suggestions: List[Suggestion]
    llm_used: bool = False
    llm_error: Optional[str] = None


@dataclass
class FaithfulnessReport:
    score: float
    checks: int
    passed: int
    issues: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    target_role: str
    coverage_score: int            # 0-100, weighted (required=1.0, preferred=0.5; partial counts half)
    demonstrated_count: int
    total_skills: int
    findings: List[SkillFinding]
    summary: str
    suggestions: List[Suggestion]
    faithfulness: FaithfulnessReport
    llm_used: bool
    llm_error: Optional[str] = None
