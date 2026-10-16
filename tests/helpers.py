from app.config import ROOT, Settings
from app.pipeline import build_pipeline

RESUME = (ROOT / "data/samples/resume_sample.txt").read_text()
JD = (ROOT / "data/samples/jd_data_scientist.txt").read_text()


def offline_pipeline():
    return build_pipeline(Settings.offline())
