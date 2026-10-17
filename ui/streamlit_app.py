"""Streamlit UI. Run: streamlit run ui/streamlit_app.py
If API_URL is set it calls the REST API, otherwise it runs the pipeline in-process."""
import dataclasses
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import ROOT, Settings  # noqa: E402
from app.ingestion.parser import ParseError, parse_document  # noqa: E402

st.set_page_config(page_title="Resume Intelligence", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}
        [data-testid="stHeader"] {background: transparent;}
        .hero {
            padding: 1.8rem 2rem; border-radius: 18px; margin-bottom: 1.4rem;
            background: linear-gradient(120deg, #102c3b 0%, #174b52 100%);
            color: #f5fbfa;
        }
        .hero h1 {font-size: clamp(1.9rem, 4vw, 2.7rem); margin: 0 0 .5rem 0; color: #fff;}
        .hero p {font-size: 1.05rem; max-width: 760px; margin: 0; color: #d4e7e4;}
        .eyebrow {
            color: #a9d7c7; text-transform: uppercase; letter-spacing: .12em;
            font-size: .76rem; font-weight: 700; margin-bottom: .55rem;
        }
        div[data-testid="stMetric"] {
            background: #f7faf9; border: 1px solid #e2ebe8; padding: 1rem 1.1rem;
            border-radius: 14px;
        }
        div[data-testid="stMetricLabel"] {color: #536773;}
        div[data-testid="stMetricValue"] {color: #153d43;}
        .section-note {color: #586b73; font-size: .94rem; margin-top: -.4rem;}
        @media (max-width: 700px) {
            .block-container {padding: 1.2rem 1rem 3rem;}
            .hero {padding: 1.35rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading analysis models (first run only)...")
def _pipeline():
    from app.pipeline import build_pipeline
    return build_pipeline(Settings())


def analyze(resume_text: str, jd: str, role: str) -> dict:
    api = os.getenv("API_URL", "").rstrip("/")
    if api:
        import requests
        response = requests.post(
            f"{api}/analyze/text",
            timeout=120,
            json={"resume_text": resume_text, "job_description": jd, "target_role": role},
        )
        response.raise_for_status()
        return response.json()
    return dataclasses.asdict(_pipeline().analyze(resume_text, jd, role))


def _render_finding(finding: dict) -> None:
    importance = finding["importance"].capitalize()
    label = f'{finding["skill"]} · {importance}'
    with st.expander(label):
        st.write(finding["reason"])
        st.caption(f'Match confidence: {finding["confidence"]:.0%}')
        if finding["evidence"]:
            st.markdown("**Resume evidence**")
            for evidence in finding["evidence"]:
                st.markdown(f'> “{evidence["text"]}”')
                source = evidence["section"].replace("_", " ").title()
                details = [source, evidence["match_type"].capitalize()]
                if evidence["context"]:
                    details.insert(0, evidence["context"])
                st.caption(" · ".join(details))
        elif finding["status"] == "missing":
            st.caption("No supporting passage was found in the resume.")


def _render_results(result: dict) -> None:
    st.divider()
    st.subheader("Your match overview")
    st.caption(
        "The match score compares the resume with skills identified in the job description. "
        "Required skills carry more weight; partially supported skills count as half."
    )
    score_col, skill_col, checks_col = st.columns(3)
    score_col.metric("Overall match", f'{result["coverage_score"]}%')
    skill_col.metric(
        "Skills demonstrated",
        f'{result["demonstrated_count"]} of {result["total_skills"]}',
    )
    faithfulness = result["faithfulness"]
    checks_col.metric(
        "Evidence checks",
        f'{faithfulness["passed"]} of {faithfulness["checks"]} passed',
    )
    st.progress(max(0, min(100, int(result["coverage_score"]))) / 100)

    st.markdown("#### Summary")
    st.write(result["summary"])
    if not result["llm_used"]:
        st.info(
            "The optional AI explanation was unavailable, so this summary was generated "
            "using the built-in fallback."
        )
        if result.get("llm_error"):
            with st.expander("Technical details"):
                st.code(result["llm_error"])

    findings = result["findings"]
    categories = [
        ("demonstrated", "Supported by experience"),
        ("partial", "Some evidence"),
        ("missing", "Not found yet"),
    ]
    tabs = st.tabs(
        [f"{title} · {sum(f['status'] == status for f in findings)}" for status, title in categories]
    )
    for tab, (status, title) in zip(tabs, categories):
        with tab:
            matched = [finding for finding in findings if finding["status"] == status]
            if not matched:
                st.success(f"No skills in “{title.lower()}” for this analysis.")
            else:
                st.caption(
                    {
                        "demonstrated": "These skills have a supporting passage in an experience or project section.",
                        "partial": "The resume mentions these skills, but stronger experience evidence may help.",
                        "missing": "These job-description skills were not found in the resume.",
                    }[status]
                )
                for finding in matched:
                    _render_finding(finding)

    if result["suggestions"]:
        st.markdown("#### Suggested next steps")
        st.caption("Use these ideas to make relevant experience easier for an employer to find.")
        for suggestion in result["suggestions"]:
            with st.container(border=True):
                st.markdown(f'**{suggestion["skill"]}**')
                st.write(suggestion["advice"])

    if faithfulness["issues"]:
        with st.expander("Evidence validation details"):
            for issue in faithfulness["issues"]:
                st.write(f"- {issue}")


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Resume Intelligence</div>
        <h1>See how your experience fits the role.</h1>
        <p>Compare a resume with a job description, understand which skills are supported,
        and review the passages behind each result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Start an analysis")
st.markdown(
    '<p class="section-note">Add a resume and the job description you are applying for. '
    "Your results will appear below.</p>",
    unsafe_allow_html=True,
)

with st.form("analysis_form"):
    use_sample = st.checkbox(
        "Try the sample resume and job description",
        help="Use the included example candidate and Junior Data Scientist role.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 1. Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, Word (.docx), and plain text (.txt).",
        )
        st.caption("Choose a file from your device, or select the sample option above.")
    with right:
        st.markdown("#### 2. Job description")
        target_role = st.text_input(
            "Role or job title (optional)",
            value="Junior Data Scientist" if use_sample else "",
            placeholder="e.g. Data Analyst",
        )
        default_jd = (
            (ROOT / "data" / "samples" / "jd_data_scientist.txt").read_text(encoding="utf-8")
            if use_sample
            else ""
        )
        job_description = st.text_area(
            "Paste the job description",
            value=default_jd,
            height=220,
            placeholder="Paste the full job description here so we can identify its skills and requirements.",
        )
    submitted = st.form_submit_button("Analyze resume", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("analysis_result", None)
    try:
        if use_sample:
            resume_text = (ROOT / "data" / "samples" / "resume_sample.txt").read_text(
                encoding="utf-8"
            )
        elif uploaded_file is not None:
            resume_text = parse_document(uploaded_file.name, uploaded_file.getvalue())
        else:
            st.error("Upload a resume, or select the sample resume option.")
            resume_text = ""

        if resume_text and len(resume_text.strip()) < 50:
            st.error("This resume does not contain enough readable text. Please try another file.")
        elif len(job_description.strip()) < 30:
            st.error("Please paste a job description with at least a few sentences.")
        elif resume_text:
            with st.spinner("Comparing skills and finding resume evidence..."):
                result = analyze(resume_text, job_description, target_role.strip())
            if result["total_skills"] == 0:
                st.error(
                    "We could not identify skills in this job description. "
                    "Try including the responsibilities and requirements sections."
                )
            else:
                st.session_state["analysis_result"] = result
    except ParseError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Analysis failed: {error}")

if "analysis_result" in st.session_state:
    _render_results(st.session_state["analysis_result"])
