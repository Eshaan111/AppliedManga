"""Diagnose saved validation TextExtractionPipeline results; never rerun joining."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from text_extraction_diagnostics import run_diagnostics


def run(candidates_path, output):
    return run_diagnostics(ROOT, candidates_path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path,
                        default=ROOT / 'outputs/text_extraction/val/candidates.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/diagnostics/ocr_joining_validation')
    args = parser.parse_args()
    try:
        summary = run(args.candidates, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(summary, indent=2))
    print('Outputs:', args.output.resolve())


if __name__ == '__main__':
    main()
