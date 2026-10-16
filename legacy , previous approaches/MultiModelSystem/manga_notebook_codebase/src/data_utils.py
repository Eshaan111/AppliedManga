from pathlib import Path
import json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def get_sequence_images(image_root, sequence_id):
    folder = Path(image_root) / sequence_id
    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if len(images) != 3:
        raise ValueError(f"{sequence_id}: expected 3 images, found {len(images)}")
    return images

def split_sequence_ids(rows, val_ratio=0.2, seed=42):
    import random
    ids = sorted(x["sequence_id"] for x in rows)
    if len(ids) != len(set(ids)) or len(ids) < 2 or not 0 < val_ratio < 1:
        raise ValueError("Need unique sequence IDs, at least two sequences, and 0 < val_ratio < 1")
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = min(len(ids) - 1, max(1, int(len(ids) * val_ratio)))
    return ids[n_val:], ids[:n_val]
