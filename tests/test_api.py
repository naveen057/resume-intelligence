import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.pipeline import build_pipeline  # noqa: E402
from tests.helpers import JD, RESUME  # noqa: E402

main.get_pipeline.cache_clear()
main.get_pipeline = lambda: build_pipeline(Settings.offline())   # no model downloads in tests
client = TestClient(main.app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_analyze_text_ok():
    r = client.post("/analyze/text", json={"resume_text": RESUME, "job_description": JD, "target_role": "DS"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["coverage_score"] <= 100 and body["findings"]
    assert body["faithfulness"]["score"] == 1.0


def test_analyze_text_validation_error():
    assert client.post("/analyze/text", json={"resume_text": "short", "job_description": JD}).status_code == 422


def test_analyze_file_upload_and_bad_type():
    ok = client.post("/analyze", files={"resume": ("cv.txt", RESUME.encode())}, data={"job_description": JD})
    assert ok.status_code == 200
    bad = client.post("/analyze", files={"resume": ("cv.exe", b"x" * 500)}, data={"job_description": JD})
    assert bad.status_code == 422


def test_jd_without_skills_is_rejected():
    r = client.post("/analyze/text", json={"resume_text": RESUME, "job_description": "We value kindness and curiosity in everyone."})
    assert r.status_code == 422
