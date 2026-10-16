"""Inspect saved inference outputs without loading models or reference labels."""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_tokens(text):
    # Include OCR confusions such as 17O, while avoiding arbitrary English words.
    words = re.findall(r"\b[0-9OoIl]+\b", text)
    return [word.translate(str.maketrans("OoIl", "0011")) for word in words if any(c.isdigit() for c in word)]


def audit(output_dir, pipeline_root):
    score_file = pipeline_root.parent / "TaskData/score.py"
    spec = importlib.util.spec_from_file_location("task_schema", score_file)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    schema = {}
    for name in ["transcript.jsonl", "transcript_raw.jsonl"]:
        try:
            rows = score.load_jsonl(output_dir / name)
            schema[name] = dict(valid=True, sequence_ids=list(rows))
        except (ValueError, OSError) as error:
            schema[name] = dict(valid=False, error=str(error))

    boxes = read(output_dir / "box_predictions.json")
    groups = read(output_dir / "group_predictions.json")
    edits = read(output_dir / "transcript_cleanup_audit.json")
    survivors = [b["box_id"] for b in boxes if not b["delete_ocr"]]
    emitted = [bid for group in groups for bid in group["box_ids"]]
    membership = dict(exact_once=len(emitted) == len(set(emitted)) and set(emitted) == set(survivors),
                      surviving_boxes=len(survivors), grouped_members=len(emitted))
    interleaved = []
    for group in groups:
        page_ids = [b["box_id"] for b in boxes if not b["delete_ocr"] and b["page_index"] == group["page_index"]]
        positions = sorted(page_ids.index(bid) for bid in group["box_ids"])
        if positions and positions[-1] - positions[0] + 1 != len(positions):
            interleaved.append(dict(group=group["box_id"], members=group["box_ids"], positions=positions))
    changed = [row for row in edits if row["changed"]]
    numeric_changes = [row for row in changed if numeric_tokens(row["raw_text"]) != numeric_tokens(row["cleaned_text"])]

    # Execute only two pure decision functions; never execute a notebook cell.
    notebook_path = pipeline_root / "pipelineArtifactContourFullInference.ipynb"
    notebook = read(notebook_path)
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            definitions.extend(node for node in ast.parse("".join(cell["source"])).body
                               if isinstance(node, ast.FunctionDef) and node.name in {"punctuation_only", "direct_decision"})
    namespace = {"unicodedata": unicodedata}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "pure_decisions", "exec"), namespace)
    cases = [("", "Readable sentence."), ("", "..."), ("I", "I"), ("0", "0"), ("~~~~~", "")]
    decisions = [dict(paddle=p, magi=m, decision=namespace["direct_decision"](p, m)) for p, m in cases]

    # This is a candidate safeguard, not a change to cleanup or a correctness classifier.
    assert numeric_tokens("17O GRAND") != numeric_tokens("TO GRAND")
    assert numeric_tokens("17O GRAND") == numeric_tokens("170 GRAND")
    assert numeric_tokens("I HAVE 2 QUESTIONS") == numeric_tokens("I HAVE 2 QUESTIONS.")
    return dict(audited_at=datetime.now(timezone.utc).isoformat(), output_dir=str(output_dir.resolve()),
                notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                reference_labels_read=False, model_inference_run=False, predictions_modified=False,
                schema=schema, membership=membership, original_boxes=len(boxes), groups=len(groups),
                deleted_boxes=[dict(box_id=b["box_id"], reason=b["deletion_reason"],
                                    paddle=b["paddle_text"], magi=b["magi_text"]) for b in boxes if b["delete_ocr"]],
                roles=dict(Counter(g["text_role"] for g in groups)),
                speaker_status=dict(Counter(g["speaker_status"] for g in groups)),
                review_groups=[g["box_id"] for g in groups if g["needs_review"]],
                tail_override_groups=sum(bool(g["tail_override"]) for g in groups),
                noncontiguous_merge_groups=interleaved, cleanup_changed_entries=len(changed),
                cleanup_numeric_change_flags=numeric_changes, direct_decision_cases=decisions,
                limits="A single saved run and synthetic checks; no accuracy claim. Numeric flags require review. Saved outputs may precede current notebook edits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="One run's outputs directory")
    parser.add_argument("--report", type=Path, help="Optional separate report file; predictions are never edited")
    args = parser.parse_args()
    report = audit(args.output_dir, Path(__file__).resolve().parents[1])
    text = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
