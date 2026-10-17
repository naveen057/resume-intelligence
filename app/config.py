"""Central configuration. Everything is overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


# Load Streamlit Cloud secrets into environment variables.
try:
    import streamlit as st

    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "USE_LLM",
        "EMBEDDER_BACKEND",
        "EMBEDDING_MODEL",
        "RERANKER_BACKEND",
        "RERANKER_MODEL",
        "SEMANTIC_THRESHOLD",
        "API_URL",
        "MAX_UPLOAD_MB",
    ):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = str(st.secrets[key])

except (ImportError, FileNotFoundError):
    pass


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Retrieval
    embedder_backend: str = field(default_factory=lambda: os.getenv("EMBEDDER_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    reranker_backend: str = field(default_factory=lambda: os.getenv("RERANKER_BACKEND", "cross-encoder"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    candidate_k: int = 8          # candidates passed to the reranker
    evidence_k: int = 3           # max evidence snippets per skill
    semantic_threshold: float = field(default_factory=lambda: float(os.getenv("SEMANTIC_THRESHOLD", "0.5")))
    # LLM
    use_llm: bool = field(default_factory=lambda: _bool("USE_LLM", "true"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # App
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", ""))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "5")))
    taxonomy_path: Path = ROOT / "data" / "skills_taxonomy.json"

    @classmethod
    def offline(cls) -> "Settings":
        """No model downloads, no network. Used by tests and CI."""
        return cls(embedder_backend="hashing", reranker_backend="none", use_llm=False, gemini_api_key="")
