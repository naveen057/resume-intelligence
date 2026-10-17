from app.analysis.llm import SafeExplainer, TemplateExplainer
from app.analysis.matcher import coverage
from app.analysis.skills import SkillExtractor
from app.config import Settings
from app.evaluation.faithfulness import check_faithfulness
from app.models import Evidence, Explanation, SkillFinding, Suggestion
from tests.helpers import JD, RESUME, offline_pipeline

EXTRACTOR = SkillExtractor.from_file(Settings().taxonomy_path)


def test_alias_matching_respects_word_boundaries():
    assert EXTRACTOR.find("Used sklearn and C++") >= {"Scikit-learn", "C++"}
    assert "Git" not in EXTRACTOR.find("Used digit and gitops")   # no substring matches
    assert "Git" in EXTRACTOR.find("Used GitHub daily")
    assert "R" not in EXTRACTOR.find("worked in R&D and r e s t")
    assert "R" in EXTRACTOR.find("Statistics in R and Python")


def test_jd_importance_sections_and_alternatives():
    skills = {s.name: s for s in EXTRACTOR.extract_jd(JD)}
    assert skills["Python"].importance == "required"
    assert skills["Tableau"].importance == "preferred"
    assert skills["RAG"].importance == "preferred"
    assert skills["Flask"].alternatives == ["FastAPI"]
    assert skills["FastAPI"].alternatives == ["Flask"]
    assert skills["REST APIs"].alternatives == []        # 'with' is not an alternative connector
    assert skills["Tableau"].alternatives == ["Power BI"] # via "Tableau or Power BI"


def test_pipeline_verdicts_on_sample():
    res = offline_pipeline().analyze(RESUME, JD, "Junior Data Scientist")
    status = {f.skill: f.status for f in res.findings}
    assert status["FastAPI"] == "demonstrated"
    assert status["Spark"] == "missing" and status["AWS"] == "missing"
    assert status["Tableau"] == "partial"            # skills-section only
    assert "Flask" not in status                     # satisfied by 'FastAPI or Flask'
    assert 0 < res.coverage_score < 100
    assert res.demonstrated_count == sum(s == "demonstrated" for s in status.values())


def test_every_demonstrated_skill_has_cited_evidence():
    res = offline_pipeline().analyze(RESUME, JD)
    for f in res.findings:
        if f.status == "demonstrated":
            assert f.evidence and f.evidence[0].section not in {"skills", "summary", "education"}
        if f.status == "missing":
            assert f.evidence == []
    assert res.faithfulness.score == 1.0


def test_coverage_weights():
    mk = lambda imp, st: SkillFinding("x", "c", imp, st, 1.0, "")
    assert coverage([mk("required", "demonstrated"), mk("required", "missing")]) == 50
    assert coverage([mk("required", "demonstrated"), mk("preferred", "missing")]) == 67
    assert coverage([mk("required", "partial")]) == 50
    assert coverage([]) == 0


def test_faithfulness_catches_hallucinated_skill_and_bad_citation():
    findings = [SkillFinding("Python", "language", "required", "demonstrated", 0.9, "",
                             [Evidence("resume-1", "Built models in Python", "experience", "", 1.0, None, "keyword")])]
    bad = Explanation("You should learn Kubernetes.", [Suggestion("Python", "Great Python work.", ["resume-99"])])
    report = check_faithfulness(findings, bad, EXTRACTOR, "Built models in Python", "Python role")
    assert report.score < 1.0 and len(report.issues) == 2
    assert any("Kubernetes" in i for i in report.issues) and any("resume-99" in i for i in report.issues)


def test_faithfulness_flags_unsupported_demonstrated_verdict():
    f = SkillFinding("Docker", "devops", "required", "demonstrated", 0.9, "",
                     [Evidence("resume-1", "Used Python", "experience", "", 1.0, None, "keyword")])
    report = check_faithfulness([f], Explanation("ok", []), EXTRACTOR, "Used Python", "Docker")
    assert report.issues


def test_llm_failure_falls_back_to_template():
    class Boom:
        def explain(self, *a):
            raise RuntimeError("quota exceeded")

    res = offline_pipeline().analyze(RESUME, JD)
    out = SafeExplainer(Boom(), TemplateExplainer()).explain("DS", res.findings, res.coverage_score)
    assert not out.llm_used and "quota exceeded" in out.llm_error and out.summary


def test_llm_initialization_failure_is_reported():
    res = offline_pipeline().analyze(RESUME, JD)
    out = SafeExplainer(None, TemplateExplainer(), "Gemini client initialization failed").explain(
        "DS", res.findings, res.coverage_score
    )
    assert not out.llm_used
    assert out.llm_error == "Gemini client initialization failed"
