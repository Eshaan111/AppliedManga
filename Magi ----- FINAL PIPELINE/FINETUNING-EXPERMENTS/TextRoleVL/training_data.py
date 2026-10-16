"""Shared data preparation for experiments A/B/C; inference notebooks stay unchanged."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
MAGI_ROOT = ROOT.parent
CACHE = MAGI_ROOT / "paddleOCR-Magi" / "cache_baseline" / "development"
DATA = ROOT / "training_data"

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

# Reuse the annotation functions from notebook 1 (same coordinates, IDs, and style).
font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 14)

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def dotted_line(draw, start, end):
    length = np.hypot(end[0] - start[0], end[1] - start[1])
    count = max(1, int(length / 6))
    for step in range(count + 1):
        t = step / count
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="purple")

def labeled_box(draw, image, box, label, color, placed_labels):
    draw.rectangle(box, outline=color, width=2)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width, height = right - left + 6, bottom - top + 4
    x = max(0, min(int(box[0]), image.width - width))
    y = max(0, min(int(box[1]) - height - 2, image.height - height))
    # Move crowded labels slightly up/down so IDs do not overlap one another.
    candidates = [y]
    for step in range(1, 9):
        candidates.extend([max(0, y - step * (height + 2)),
                           min(image.height - height, y + step * (height + 2))])
    for candidate in candidates:
        bounds = (x, candidate, x + width, candidate + height)
        if all(bounds[2] <= a or bounds[0] >= c or bounds[3] <= b or bounds[1] >= d
               for a, b, c, d in placed_labels):
            y = candidate
            break
    placed_labels.append((x, y, x + width, y + height))
    draw.rectangle((x, y, x + width, y + height), fill="white", outline=color)
    draw.text((x + 3 - left, y + 2 - top), label, font=font, fill=color)


def prepare_training_data():
    labels = {}
    path = MAGI_ROOT / "TaskData" / "development" / "labels.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["sequence_id"]] = row
    assert len(labels) == 80, "Expected the audited 80 sequences; review the split if data changes."
    examples, exclusions, source_hashes = [], [], {}
    for sid, gt in sorted(labels.items()):
        folder = CACHE / sid
        raw_saved = read_json(folder / "magi_raw.json")
        match_saved = read_json(folder / "gt_matches.json")
        assert digest(raw_saved["data"]) == raw_saved["data_hash"]
        assert digest(match_saved["data"]) == match_saved["data_hash"]
        assert match_saved["settings"]["magi_data_hash"] == raw_saved["data_hash"]
        source_hashes[sid] = [raw_saved["data_hash"], match_saved["data_hash"], digest(gt)]
        pages = raw_saved["data"]
        assert len(pages) == len(gt["pages"]) == 3
        matches = {(r["page_index"], r["text_index"]): r
                   for r in match_saved["data"]["bubble_matches"]}
        assert len(matches) == sum(len(p["texts"]) for p in pages)
        manifest = {"sequence_id": sid, "texts": [], "tails": [], "text_tail_links": []}
        included, page_hashes, image_paths = [], [], []
        offset = 0
        output_folder = DATA / "pages" / sid
        output_folder.mkdir(parents=True, exist_ok=True)
        for p, page in enumerate(pages):
            original = MAGI_ROOT / "TaskData" / "development" / "images" / sid / f"{p+1:02d}.png"
            page_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            assert page_hash == raw_saved["settings"]["pages"][p]["sha256"]
            page_hashes.append(page_hash)
            with Image.open(original) as image:
                image = image.convert("L").convert("RGB")
            draw, placed = ImageDraw.Draw(image), []
            tails = page.get("tails", [])
            for ti, tail_i in page.get("text_tail_associations", []):
                assert 0 <= ti < len(page["texts"]) and 0 <= tail_i < len(tails)
                dotted_line(draw, center(page["texts"][ti]), center(tails[tail_i]))
                manifest["text_tail_links"].append({"text_id": f"PG{p+1}:T{ti+1}",
                                                   "tail_id": f"PG{p+1}:TAIL{tail_i+1}"})
            for i, box in enumerate(tails):
                tail_id = f"PG{p+1}:TAIL{i+1}"
                labeled_box(draw, image, box, tail_id, "purple", placed)
                manifest["tails"].append({"id": tail_id, "page": p+1})
            assert len(page["texts"]) == len(page["ocr"])
            for i, (box, ocr) in enumerate(zip(page["texts"], page["ocr"])):
                text_id = f"PG{p+1}:T{i+1}"
                labeled_box(draw, image, box, text_id, "red", placed)
                manifest["texts"].append({"id": text_id, "page": p+1, "text": ocr})
                match = matches[(p, i)]
                assert np.allclose(box, match["box"], atol=0.01, rtol=0)
                assert ocr == match["ocr_text"]
                speaker = match.get("gt_speaker")
                reason = None
                if not match.get("eligible_for_training") or match.get("needs_review"):
                    reason = "unmatched_or_review_needed"
                elif speaker != "NARRATION" and not re.fullmatch(r"character[1-9][0-9]*", speaker or ""):
                    reason = "UNKNOWN_or_invalid_GT_speaker"
                if reason:
                    exclusions.append({"sequence_id": sid, "id": text_id, "reason": reason,
                                       "match_stage": match["stage"], "ocr": ocr})
                    continue
                gt_index = match["gt_index"]
                actual_gt = gt["pages"][p][gt_index]
                assert actual_gt["speaker"] == speaker and actual_gt["text"] == match["gt_text"]
                included.append({"id": text_id, "role": "NARRATION" if speaker == "NARRATION" else "SPOKEN",
                                 "reading_position": offset + gt_index, "text_index": i,
                                 "match_stage": match["stage"]})
            offset += len(gt["pages"][p])
            annotated = output_folder / f"annotated_page_{p+1}.png"
            image.save(annotated)
            image_paths.append(str(annotated))
        # Whole examples; never shuffle or randomly select regions inside a sequence.
        included.sort(key=lambda r: (r["reading_position"], r["text_index"]))
        targets = [{"id": r["id"], "role": r["role"]} for r in included]
        transitions = []
        for left, right in zip(included, included[1:]):
            if (right["reading_position"] == left["reading_position"] + 1
                    and left["role"] == "NARRATION" and right["role"] == "SPOKEN"):
                transitions.append([left["id"], right["id"]])
        examples.append({"sequence_id": sid, "images": image_paths, "page_hashes": page_hashes,
                         "manifest": manifest, "target": targets, "transitions": transitions,
                         "has_narration_gt": any(t["speaker"] == "NARRATION" for pg in gt["pages"] for t in pg),
                         "has_narration_target": any(t["role"] == "NARRATION" for t in targets)})
        write_json(output_folder / "manifest.json", manifest)

    # One fixed stratified sequence split shared by all experiments.
    rng = random.Random(42)
    positives = [e["sequence_id"] for e in examples if e["has_narration_gt"]]
    negatives = [e["sequence_id"] for e in examples if not e["has_narration_gt"]]
    assert len(positives) == 14
    rng.shuffle(positives)
    rng.shuffle(negatives)
    validation_ids = set(positives[:3] + negatives[:13])
    train_ids = set(labels) - validation_ids
    split = {"train": sorted(train_ids), "validation": sorted(validation_ids),
             "seed": 42, "source_hashes": source_hashes}
    split_path = DATA / "split.json"
    if split_path.exists():
        assert read_json(split_path) == split, "Data changed; preserve old runs and review the shared split."
    else:
        write_json(split_path, split)
    assert len(train_ids) == 64 and len(validation_ids) == 16
    assert not ({h for e in examples if e["sequence_id"] in train_ids for h in e["page_hashes"]}
                & {h for e in examples if e["sequence_id"] in validation_ids for h in e["page_hashes"]})
    for name, ids in [("train", train_ids), ("validation", validation_ids)]:
        assert any(e["has_narration_target"] for e in examples if e["sequence_id"] in ids)
        for e in examples:
            if e["sequence_id"] in ids:
                e["split"] = name
    write_json(DATA / "examples.json", examples)
    write_json(DATA / "exclusions.json", exclusions)
    for e in examples:
        if not e["target"]:
            exclusions.append({"sequence_id": e["sequence_id"], "id": None,
                               "reason": "sequence_has_no_resolved_targets"})
    write_json(DATA / "exclusions.json", exclusions)
    print("Fixed split: 64 train / 16 validation; narration-containing GT sequences: 11 / 3.")
    for name in ["train", "validation"]:
        usable = [e for e in examples if e["split"] == name and e["target"]]
        counts = Counter(t["role"] for e in usable for t in e["target"])
        print(name, "usable sequences:", len(usable), "resolved roles:", dict(counts))
    print("Exclusions:", dict(Counter(r["reason"] for r in exclusions)))
    return examples
