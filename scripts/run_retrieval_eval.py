"""CLI: score a job/collection against labeled questions (Day 2 Precision@K)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.config import get_settings
from clinical_rag.eval.runner import load_questions, run_retrieval_eval
from clinical_rag.errors import IngestError


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run retrieval eval for a job combination")
    p.add_argument("--job-id", required=True)
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    p.add_argument("--k", default="3,5,10", help="Comma-separated k values")
    p.add_argument("--eval-set-id", default="default")
    return p.parse_args()


def main() -> None:
    args = _args()
    settings = get_settings()
    report_path = Path(settings.jobs_dir) / args.job_id / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"Job report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    try:
        questions = load_questions(args.questions)
        result = run_retrieval_eval(
            job_report=report,
            questions=questions,
            persist_dir=settings.chroma.persist_dir,
            k_values=k_values,
            evals_dir=Path("artifacts/evals"),
            eval_set_id=args.eval_set_id,
        )
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.metrics, indent=2))
    print(f"Wrote {result.run_dir}")


if __name__ == "__main__":
    main()
