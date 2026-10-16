"""Run: python -m app.evaluation.run_eval   (set EMBEDDER_BACKEND=hashing to run offline)"""
from __future__ import annotations

import json

from app.config import ROOT, Settings
from app.evaluation.metrics import evaluate_retrieval
from app.pipeline import build_pipeline
from app.retrieval.embedder import get_embedder


def main() -> None:
    settings = Settings()
    print(f"Embedder: {settings.embedder_backend} ({settings.embedding_model})\n")
    results = evaluate_retrieval(ROOT / "data/eval/retrieval_eval.json", get_embedder(settings))

    cols = list(next(iter(results.values())))
    print(f"{'mode':<10}" + "".join(f"{c:>12}" for c in cols))
    for mode, row in results.items():
        print(f"{mode:<10}" + "".join(f"{row[c]:>12.3f}" for c in cols))

    pipe = build_pipeline(settings)
    resume = (ROOT / "data/samples/resume_sample.txt").read_text()
    jd = (ROOT / "data/samples/jd_data_scientist.txt").read_text()
    res = pipe.analyze(resume, jd, "Junior Data Scientist")
    print(f"\nEnd-to-end faithfulness: {res.faithfulness.score} "
          f"({res.faithfulness.passed}/{res.faithfulness.checks} checks, llm_used={res.llm_used})")
    out = ROOT / "eval_results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps({"retrieval": results,
                                                 "faithfulness": res.faithfulness.score}, indent=2))
    print(f"Saved to {out / 'latest.json'}")


if __name__ == "__main__":
    main()
