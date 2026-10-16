from app.ingestion.chunker import chunk_text
from app.ingestion.parser import ParseError, clean_text, parse_document
from tests.helpers import RESUME


def test_bullets_become_individual_chunks_with_sections():
    chunks = chunk_text(RESUME)
    fastapi = [c for c in chunks if "FastAPI" in c.text]
    assert len(fastapi) == 1
    assert fastapi[0].section == "experience"
    assert fastapi[0].context.startswith("Data Science Intern")


def test_skills_lines_are_kept_separately():
    skills = [c for c in chunk_text(RESUME) if c.section == "skills"]
    assert len(skills) == 3 and any("Python, SQL" in c.text for c in skills)


def test_chunk_ids_are_unique_and_prefixed():
    chunks = chunk_text(RESUME, source="r1")
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.id.startswith("r1-") for c in chunks)


def test_long_paragraph_is_split():
    text = "EXPERIENCE\n" + " ".join(f"Sentence number {i} is here." for i in range(60))
    assert len(chunk_text(text, max_words=40)) > 1


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\nb\r\nc") == "a\n\nb\nc"


def test_parse_txt_and_reject_bad_types():
    assert "Aarav" in parse_document("cv.txt", RESUME.encode())
    for name, data in [("cv.exe", b"x" * 500), ("cv.txt", b"too short")]:
        try:
            parse_document(name, data)
            raise AssertionError("expected ParseError")
        except ParseError:
            pass


def test_parse_docx_roundtrip():
    import io
    from docx import Document
    doc = Document()
    for line in RESUME.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    assert "FastAPI" in parse_document("cv.docx", buf.getvalue())
