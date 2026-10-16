"""Inventory unmatched diagnostic groups for visual review, without labeling gold."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_exclusions import exclusion_hints, load_lexicon


def build_review(allocations, output):
    lexicon = load_lexicon()
    rows = []
    with Path(allocations).open(encoding="utf-8") as handle:
        for line in handle:
            page = json.loads(line)
            for group in page["groups"]:
                if group["target_index"] is not None:
                    continue
                rows.append(dict(sequence_id=page["sequence_id"], page_index=page["page_index"],
                    group_id=group["id"], candidate_ids=group["candidate_ids"],
                    text=group["text"], box=group["box"],
                    hints=exclusion_hints(group["text"], lexicon), keep=None,
                    exclusion_category=None, notes="", source="unmatched_diagnostic_group"))
    # Exclusive creation protects any manual review decisions on a rerun.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocations", type=Path,
        default=ROOT / "outputs/diagnostics/ocr_joining/allocations.jsonl")
    parser.add_argument("--output", type=Path,
        default=ROOT / "datasets/annotations/ocr_exclusion_review.jsonl")
    args = parser.parse_args()
    print(f"Wrote {build_review(args.allocations, args.output)} unreviewed groups to {args.output}")
