"""Triage Desk CLI.

  python triage.py generate                 # (re)create the synthetic dataset in data/
  python triage.py run [--mode auto|ai|rules]
  python triage.py evaluate                 # rules mode always; AI mode too if ANTHROPIC_API_KEY is set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_messages(path: Path):
    from models import Message
    return [Message(**m) for m in json.loads(path.read_text(encoding="utf-8"))]


def make_llm(mode: str):
    from llm import LLM, ai_available
    if mode == "rules":
        return None
    if ai_available():
        return LLM()
    if mode == "ai":
        sys.exit("ANTHROPIC_API_KEY is not set. Use --mode rules, or set the key.")
    print("No ANTHROPIC_API_KEY found: running rule-based mode (keywords + TF-IDF).")
    return None


def cmd_run(args):
    import export
    import pipeline
    result = pipeline.run(load_messages(Path(args.input)), make_llm(args.mode))
    paths = export.write_all(result, Path(args.out))
    print()
    print(export.daily_summary(result))
    for k, p in paths.items():
        print(f"  {k:8} {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    print("\nOpen out/review.html in a browser to review, edit and approve issues.")


def cmd_evaluate(args):
    import evaluate
    import export
    import pipeline
    from llm import ai_available
    msgs = load_messages(DATA / "messages.json")
    truth = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))
    planted = json.loads((DATA / "planted_bugs.json").read_text(encoding="utf-8"))
    quiet = lambda *_: None  # noqa: E731
    evals, skipped = [], None
    rules_result = pipeline.run(msgs, None, log=quiet)
    evals.append(evaluate.evaluate(rules_result, truth, planted))
    if ai_available() and args.mode != "rules":
        evals.append(evaluate.evaluate(pipeline.run(msgs, make_llm("ai"), log=quiet), truth, planted))
    else:
        skipped = "AI mode not run: ANTHROPIC_API_KEY not set."
    text = evaluate.report(evals, skipped)
    OUT.mkdir(exist_ok=True)
    (OUT / "eval_report.md").write_text(text, encoding="utf-8")
    (OUT / "eval_results.json").write_text(json.dumps(evals, indent=2), encoding="utf-8")
    print(text)
    print("Saved out/eval_report.md")


def main(argv=None):
    p = argparse.ArgumentParser(prog="triage", description="Triage Desk: player bug reports -> Jira-ready tickets")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="create the synthetic dataset")
    r = sub.add_parser("run", help="run the triage pipeline")
    r.add_argument("--mode", choices=["auto", "ai", "rules"], default="auto")
    r.add_argument("--input", default=str(DATA / "messages.json"))
    r.add_argument("--out", default=str(OUT))
    e = sub.add_parser("evaluate", help="score against ground truth")
    e.add_argument("--mode", choices=["both", "rules"], default="both")
    args = p.parse_args(argv)
    if args.cmd == "generate":
        import generate_data
        generate_data.main()
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
