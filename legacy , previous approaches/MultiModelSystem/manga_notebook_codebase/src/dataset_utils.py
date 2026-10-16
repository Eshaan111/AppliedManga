"""Portable sequence manifests and reproducible, leakage-free datasets."""
import hashlib
import json
import random
from pathlib import Path

from data_utils import load_jsonl, save_jsonl, split_sequence_ids
from prompts import BASELINE_PROMPT


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_datasets(root, seed=42, val_ratio=0.2):
    root = Path(root)
    task = root / "TaskData"
    labels = load_jsonl(task / "development/labels.jsonl")
    manifest = json.loads((task / "sequences.json").read_text(encoding="utf-8"))
    from eval_utils import validate_prediction_schema
    if not all(validate_prediction_schema(row) for row in labels):
        raise ValueError("Invalid development label schema")
    entries = {x["sequence_id"]: x for x in manifest}
    if len(entries) != len(manifest):
        raise ValueError("Duplicate manifest IDs")
    for entry in manifest:
        if len(entry["images"]) != 3:
            raise ValueError("Every manifest entry must have exactly three ordered images")
        for relative in entry["images"]:
            path = (task / relative).resolve()
            if not path.is_relative_to(task.resolve()) or not path.is_file():
                raise ValueError(f"Missing or unsafe image: {relative}")
    dev_ids = {x["sequence_id"] for x in labels}
    if dev_ids != {x["sequence_id"] for x in manifest if x["split"] == "development"}:
        raise ValueError("Labels and development manifest disagree")
    fingerprint = hashlib.sha256((task / "development/labels.jsonl").read_bytes()).hexdigest()
    split_path = root / "outputs/split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if (split.get("labels_sha256") != fingerprint or split.get("seed") != seed
                or split.get("val_ratio") != val_ratio):
            raise ValueError("Existing split differs. Archive outputs before deliberately creating a new split.")
    else:
        train, val = split_sequence_ids(labels, val_ratio, seed)
        split = dict(train=train, val=val, seed=seed, val_ratio=val_ratio, labels_sha256=fingerprint)
        save_json(split, split_path)
    if (set(split["train"]) & set(split["val"]) or
            set(split["train"]) | set(split["val"]) != dev_ids or
            len(split["train"]) + len(split["val"]) != len(dev_ids)):
        raise ValueError("Split overlaps, duplicates, or omits development sequences")
    by_id = {x["sequence_id"]: x for x in labels}
    out = root / "datasets"
    for name, ids in (("train", split["train"]), ("val", split["val"])):
        refs = [by_id[sid] for sid in ids]
        save_jsonl(refs, out / f"{name}_references.jsonl")
        records = [dict(sequence_id=sid, images=entries[sid]["images"],
                        prompt=BASELINE_PROMPT,
                        target=json.dumps({"pages": by_id[sid]["pages"]}, ensure_ascii=False)) for sid in ids]
        save_jsonl(records, out / f"{name}.jsonl")
    test = [dict(sequence_id=x["sequence_id"], images=x["images"], prompt=BASELINE_PROMPT)
            for x in manifest if x["split"] == "test"]
    save_jsonl(test, out / "test.jsonl")
    tasks = {"transcription": [], "reading_order": [], "speaker": []}
    crop_rows = []
    for sid in split["val"]:
        ref = by_id[sid]
        images = entries[sid]["images"]
        tasks["speaker"].append(dict(sequence_id=sid, images=images,
            text_pages=[[x["text"] for x in page] for page in ref["pages"]],
            expected=[[x["speaker"] for x in page] for page in ref["pages"]]))
        for page_index, page in enumerate(ref["pages"]):
            common = dict(sequence_id=sid, page_index=page_index, image=images[page_index])
            tasks["transcription"].append(dict(**common, expected=[x["text"] for x in page]))
            # Candidate IDs are random, not reference-order indices: no answer leakage.
            shuffled = list(enumerate(page))
            random.Random(f"{seed}:{sid}:{page_index}").shuffle(shuffled)
            candidates = [dict(id=f"t{i}", text=line["text"]) for i, (_, line) in enumerate(shuffled)]
            original_to_id = {original: f"t{i}" for i, (original, _) in enumerate(shuffled)}
            tasks["reading_order"].append(dict(**common, candidates=candidates,
                expected=[original_to_id[i] for i in range(len(page))]))
            for line_index, line in enumerate(page):
                if line["speaker"] not in {"UNKNOWN", "NARRATION"}:
                    crop_rows.append(dict(**common, line_index=line_index, speaker=line["speaker"],
                                          crop_id=f"{sid}_p{page_index}_l{line_index}", box=None))
    for name, rows in tasks.items():
        save_jsonl(rows, out / "components" / f"{name}.jsonl")
    annotation = out / "annotations/character_crops.jsonl"
    if not annotation.exists():
        save_jsonl(crop_rows, annotation)
    pair_path = out / "annotations/identity_pairs.jsonl"
    if not pair_path.exists():
        save_jsonl([], pair_path)
    save_json(dict(train=len(split["train"]), val=len(split["val"]), test=len(test),
                   image_count=3 * len(manifest), seed=seed, labels_sha256=fingerprint), out / "summary.json")
    return split


def read_dataset(root, split):
    return load_jsonl(Path(root) / "datasets" / f"{split}.jsonl")


def image_paths(root, row):
    return [Path(root) / "TaskData" / p for p in row["images"]]
