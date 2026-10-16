import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--task-root", required=True)
p.add_argument("--predictions", required=True)
p.add_argument("--output", default="scores.json")
p.add_argument("--references", help="Explicit scorer reference JSONL; defaults to the persisted validation dataset")
args = p.parse_args()

task_root = Path(args.task_root)
references = Path(args.references) if args.references else task_root.parent / "datasets/val_references.jsonl"
if not references.is_file():
    p.error("Prepare datasets first or supply --references")
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    str(task_root / "score.py"),
    "--references",
    str(references),
    "--predictions",
    args.predictions,
    "--output",
    args.output,
]

print(" ".join(cmd))
subprocess.run(cmd, check=True)
