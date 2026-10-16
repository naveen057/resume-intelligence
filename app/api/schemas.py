"""Pydantic models: request validation and response contract for the REST API."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    resume_text: str = Field(min_length=50, max_length=50_000)
    job_description: str = Field(min_length=30, max_length=20_000)
    target_role: str = Field(default="", max_length=120)


class EvidenceOut(BaseModel):
    chunk_id: str
    text: str
    section: str
    context: str
    retrieval_score: float
    rerank_score: Optional[float]
    match_type: Literal["keyword", "semantic"]


class SkillFindingOut(BaseModel):
    skill: str
    category: str
    importance: Literal["required", "preferred"]
    status: Literal["demonstrated", "partial", "missing"]
    confidence: float
    reason: str
    evidence: List[EvidenceOut]


class SuggestionOut(BaseModel):
    skill: str
    advice: str
    evidence_ids: List[str]


class FaithfulnessOut(BaseModel):
    score: float
    checks: int
    passed: int
    issues: List[str]


class AnalysisResponse(BaseModel):
    target_role: str
    coverage_score: int
    demonstrated_count: int
    total_skills: int
    findings: List[SkillFindingOut]
    summary: str
    suggestions: List[SuggestionOut]
    faithfulness: FaithfulnessOut
    llm_used: bool
    llm_error: Optional[str] = None
