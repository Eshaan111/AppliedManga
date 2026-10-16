"""Produce standalone JOINED OCR outputs from images/<sequence_id>/ and labels.jsonl."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from directory_ocr import run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing images/ and labels.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Dedicated output directory for this input")
    parser.add_argument("--candidates", type=Path, help="Existing candidates.jsonl; join covered inputs without running OCR")
    parser.add_argument("--limit", type=int, help="First N sequences; omit for all labelled sequences")
    parser.add_argument("--reviews", type=Path, help="Optional confirmed visual SFX review JSONL for this input")
    args = parser.parse_args()
    try:
        summary = run_directory(args.input, args.output, limit=args.limit, reviews_path=args.reviews,
                                candidates_path=args.candidates)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
