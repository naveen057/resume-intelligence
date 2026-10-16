"""FastAPI service. Run: uvicorn app.api.main:app --reload"""
from __future__ import annotations

import dataclasses
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.schemas import AnalysisResponse, AnalyzeTextRequest
from app.config import Settings
from app.ingestion.parser import ParseError, parse_document
from app.pipeline import AnalysisPipeline, build_pipeline

app = FastAPI(title="AI Resume Intelligence & Job Matching", version="1.0.0")


@lru_cache(maxsize=1)
def get_pipeline() -> AnalysisPipeline:
    """Models load once per process, on first use."""
    return build_pipeline(Settings())


def _run(resume_text: str, jd: str, role: str) -> AnalysisResponse:
    result = get_pipeline().analyze(resume_text, jd, role)
    if result.total_skills == 0:
        raise HTTPException(422, "No recognisable skills found in the job description.")
    return AnalysisResponse(**dataclasses.asdict(result))


@app.get("/health")
def health() -> dict:
    s = Settings()
    return {"status": "ok", "embedder": s.embedder_backend, "reranker": s.reranker_backend,
            "llm_configured": bool(s.use_llm and s.gemini_api_key)}


@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: AnalyzeTextRequest) -> AnalysisResponse:
    return _run(req.resume_text, req.job_description, req.target_role)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(resume: UploadFile = File(...), job_description: str = Form(..., min_length=30),
                       target_role: str = Form("")) -> AnalysisResponse:
    limit = Settings().max_upload_mb * 1024 * 1024
    data = await resume.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {Settings().max_upload_mb} MB")
    try:
        text = parse_document(resume.filename or "resume.txt", data)
    except ParseError as e:
        raise HTTPException(422, str(e))
    return _run(text, job_description, target_role)
