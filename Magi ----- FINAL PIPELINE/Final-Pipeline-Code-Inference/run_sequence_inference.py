"""Run the artifact + contour pipeline on a split from TaskData/sequences.json.

Examples:
    python run_sequence_inference.py --split test
    python run_sequence_inference.py --split development --limit 1
    python run_sequence_inference.py --split development --dry-run
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
MAGI_ROOT = PIPELINE_ROOT.parent
WORKER = PIPELINE_ROOT / "src/artifact_contour_sequence.py"
SCHEDULER = PIPELINE_ROOT / "src/artifact_contour_batch.py"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_sequences(task_data, split, selected_ids=None, limit=None):
    # Only the image manifest is read here; reference labels are never inference input.
    manifest = json.loads((task_data / "sequences.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("sequences.json must contain a list")
    records = [row for row in manifest if row.get("split") == split]
    ids = [row.get("sequence_id") for row in records]
    if any(not isinstance(sid, str) or not sid.strip() or sid != sid.strip() or Path(sid).name != sid or '/' in sid or '\\' in sid or sid in {'.','..'} for sid in ids):
        raise ValueError("Each sequence_id must be a trimmed folder name")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sequence IDs in selected split")
    if selected_ids:
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Repeated --sequence-ids")
        missing = set(selected_ids) - set(ids)
        if missing:
            raise ValueError(f"Unknown sequence IDs for {split}: {sorted(missing)}")
        records = [row for row in records if row["sequence_id"] in selected_ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError(f"No sequences selected for {split}")
    output = []
    for row in records:
        if not isinstance(row.get("images"), list) or len(row["images"]) != 3:
            raise ValueError(f"{row['sequence_id']}: expected three ordered image paths")
        images = [(task_data / path).resolve() for path in row["images"]]
        if len(set(images)) != 3:
            raise ValueError(f"{row['sequence_id']}: repeated page path")
        for path in images:
            if not path.is_file():
                raise FileNotFoundError(path)
        output.append(dict(sequence_id=row["sequence_id"], split=split, images=[str(path) for path in images]))
    return output


def load_scorer():
    # Reuse only the official schema loader, without loading references or scoring.
    spec = importlib.util.spec_from_file_location("task_schema", MAGI_ROOT / "TaskData/score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_shared_stages(settings, output_dir):
    for config in settings:
        sequence_dir = Path(config["sequence_output_dir"])
        write_json(sequence_dir / "inference_config.json", config)
        (sequence_dir / "inference.log").write_text("", encoding="utf-8")
    job = output_dir / "batch_job.json"
    stage_status = output_dir / "stage_status.json"
    write_json(job, dict(sequences=settings, status_path=str(stage_status)))
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # One process for the entire batch: models are reused at each stage boundary.
    result = subprocess.run([sys.executable, str(SCHEDULER), str(job)],
                            cwd=str(PIPELINE_ROOT), env=environment)
    if result.returncode:
        details = json.loads(stage_status.read_text(encoding="utf-8")) if stage_status.exists() else {}
        raise RuntimeError(f"Batch stage failed: {details.get('phase')}, {details.get('sequence_id')}. "
                           f"See {stage_status} and the sequence inference.log")


def run_batch(args):
    task_data = args.task_data.resolve()
    sequences = load_sequences(task_data, args.split, args.sequence_ids, args.limit)
    output_dir = (args.output_dir or MAGI_ROOT / "submission_outputs" / f"inference_{args.split}").resolve()
    cache_root = (args.cache_root or PIPELINE_ROOT / "cache_batch_artifact_contour").resolve()
    print(f"Selected {len(sequences)} {args.split} sequences, three pages each.", flush=True)
    print(f"Final predictions: {output_dir / 'predictions.jsonl'}", flush=True)
    if args.dry_run:
        for row in sequences:
            print(row["sequence_id"], "|", " | ".join(row["images"]))
        print("Dry run complete: no model loading, inference or output changes.")
        return None
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite:
        raise FileExistsError(f"{predictions_path} exists; use --overwrite to replace this batch export")
    resources = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else {}
    allowed = {"magi_model", "gemma_model", "bubble_model", "character_checkpoint", "paddle_detector", "paddle_recognizer"}
    if not isinstance(resources, dict) or set(resources) - allowed:
        raise ValueError(f"Model config accepts only: {sorted(allowed)}")
    if any(not isinstance(value, str) or not value.strip() for value in resources.values()):
        raise ValueError("Model config values must be nonempty path strings")
    resources = {key: str(Path(value).expanduser().resolve()) for key, value in resources.items()}
    scorer = load_scorer()
    status_path = output_dir / "batch_status.json"
    status = dict(started_at=datetime.now(timezone.utc).isoformat(), split=args.split,
                  manifest=str(task_data / "sequences.json"), selected_ids=[row["sequence_id"] for row in sequences],
                  worker_sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest(),
                  scheduler_sha256=hashlib.sha256(SCHEDULER.read_bytes()).hexdigest(),
                  execution="shared_models_by_stage",
                  helper_sha256={name: hashlib.sha256((PIPELINE_ROOT / name).read_bytes()).hexdigest()
                                 for name in ["contour_narrator2.py", "contour_grouping.py"]},
                  reference_labels_read=False, scoring_run=False,
                  status="running", completed=[], failed_sequence=None)
    write_json(status_path, status)
    settings = []
    for sequence in sequences:
        sequence_dir = output_dir / "sequences" / sequence["sequence_id"]
        settings.append(dict(resources, **sequence, magi_root=str(MAGI_ROOT), cache_root=str(cache_root),
                             sequence_output_dir=str(sequence_dir), clean_transcript=not args.no_cleanup, force=args.force))
    try:
        run_shared_stages(settings, output_dir)
    except Exception as error:
        stage_file = output_dir / "stage_status.json"
        detail = json.loads(stage_file.read_text(encoding="utf-8")) if stage_file.exists() else {}
        status.update(status="failed", failed_sequence=detail.get("sequence_id"),
                      failed_phase=detail.get("phase"), error=str(error))
        write_json(status_path, status)
        raise
    transcripts = []
    for config in settings:
        sid = config["sequence_id"]
        try:
            file = Path(config["sequence_output_dir"]) / "transcript.jsonl"
            parsed = scorer.load_jsonl(file, pages=3)
            if set(parsed) != {sid}:
                raise ValueError(f"Worker output sequence ID mismatch: {sorted(parsed)}")
            transcripts.append(dict(sequence_id=sid, pages=parsed[sid]))
        except Exception as error:
            status.update(status="failed", failed_sequence=sid, error=str(error))
            write_json(status_path, status)
            raise
        status["completed"].append(sid)
        write_json(status_path, status)
        print(f"Validated: {sid}", flush=True)
    # Publish the combined file only after every selected sequence succeeded.
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in transcripts), encoding="utf-8")
    checked = scorer.load_jsonl(temporary, pages=3)
    if set(checked) != set(status["selected_ids"]):
        raise ValueError("Combined export has missing or extra sequences")
    temporary.replace(predictions_path)
    status.update(status="complete", predictions=str(predictions_path),
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    print(f"Complete: {len(transcripts)} sequences -> {predictions_path}")
    print("Reference labels were not read. Run TaskData/score.py separately when references are available.")
    return predictions_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test", "development"], required=True)
    parser.add_argument("--task-data", type=Path, default=MAGI_ROOT / "TaskData")
    parser.add_argument("--sequence-ids", nargs="+", help="Optional subset; default is the whole split")
    parser.add_argument("--limit", type=int, help="Optional small first run, in manifest order")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--model-config", type=Path, help="Optional JSON with local model/checkpoint paths")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip final Gemma correction; artifact classification still runs")
    parser.add_argument("--force", action="store_true", help="Regenerate this batch runner's own stage caches")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing combined predictions.jsonl after successful inference")
    parser.add_argument("--dry-run", action="store_true", help="Check selected image paths without loading models or writing outputs")
    args = parser.parse_args()
    try:
        run_batch(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
