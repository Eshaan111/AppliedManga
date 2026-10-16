import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
MODEL_PATH = Path(r"C:\MODELS\magiv2")
SEED = 42


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_image(path):
    with Image.open(path) as image:
        return np.array(image.convert("L").convert("RGB"))


def center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def panel_id(bbox, frames):
    x, y = center(bbox)
    containing = [f for f in frames if f["bbox"][0] <= x <= f["bbox"][2] and f["bbox"][1] <= y <= f["bbox"][3]]
    if not containing:
        return None
    smallest = min(containing, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    return smallest["id"]


def pair_metadata(page, text_id, character_id):
    text = page["texts"][text_id]
    character = page["characters"][character_id]
    distances = {cid: float(np.linalg.norm(center(text["bbox"]) - center(c["bbox"]))) for cid, c in page["characters"].items()}
    nearest = min(distances, key=lambda cid: (distances[cid], cid))
    text_panel = panel_id(text["bbox"], page["frames"])
    speaker = page["speaker_links"][text_id]
    return {
        "text_id": text_id, "character_id": character_id,
        "target": int(character_id == speaker), "is_true_speaker": character_id == speaker,
        "distance": distances[character_id],
        "normalized_distance": distances[character_id] / math.hypot(page["width"], page["height"]),
        "same_panel": text_panel is not None and text_panel == panel_id(character["bbox"], page["frames"]),
        "is_nearest_character": character_id == nearest,
    }


def load_pages(base_dir, dialog_dir, image_dir):
    base_files = {p.stem: p for p in Path(base_dir).glob("*.xml")}
    dialog_files = {p.stem: p for p in Path(dialog_dir).glob("*.xml")}
    assert base_files and dialog_files, "Set the XML directory paths in notebook 01."
    missing = sorted(set(dialog_files) - set(base_files))
    assert not missing, f"Dialog books missing base XML: {missing}"
    pages, issues, speaker_types = [], [], Counter()
    for book in sorted(dialog_files):
        base = ET.parse(base_files[book]).getroot()
        dialog = ET.parse(dialog_files[book]).getroot()
        assert base.attrib["title"] == dialog.attrib["title"], book
        base_pages = {int(p.attrib["index"]): p for p in base.findall("./pages/page")}
        dialog_pages = {int(p.attrib["index"]): p for p in dialog.findall("./pages/page")}
        for index, dp in dialog_pages.items():
            assert index in base_pages, (book, index)
            bp = base_pages[index]
            width, height = int(bp.attrib["width"]), int(bp.attrib["height"])
            assert width == int(dp.attrib["width"]) and height == int(dp.attrib["height"])
            objects = {}
            for node in bp:
                if node.tag not in {"text", "body", "face", "frame"}:
                    continue
                bbox = [int(node.attrib[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                assert 0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height, (book, index, node.attrib)
                oid = node.attrib["id"]
                assert oid not in objects
                objects[oid] = {"id": oid, "bbox": bbox, "kind": node.tag}
            relations = defaultdict(set)
            for relation in dp.findall("speaker_to_text"):
                tid, sid = relation.attrib["text_id"], relation.attrib["speaker_id"]
                speaker_types[objects[sid]["kind"] if sid in objects else "UNRESOLVED_OR_OFF_PAGE"] += 1
                if tid not in objects or objects[tid]["kind"] != "text" or sid not in objects:
                    issues.append({"book": book, "page": index, "text_id": tid, "speaker_id": sid, "reason": "unresolved_or_off_page"})
                    relations[tid].add(None)
                    continue
                relations[tid].add(sid)
            # This experiment uses visible body appearances, never book-level character identities.
            unexpected = {objects[sid]["kind"] for ids in relations.values() for sid in ids if sid is not None} - {"body"}
            assert not unexpected, f"Inspect speaker semantics before proceeding: {unexpected}"
            links = {}
            for tid, ids in relations.items():
                if len(ids) == 1 and None not in ids:
                    links[tid] = next(iter(ids))
                else:
                    issues.append({"book": book, "page": index, "text_id": tid, "reason": "ambiguous_or_unresolved_relation"})
            images = [Path(image_dir) / book / f"{index:03d}{suffix}" for suffix in (".jpg", ".png", ".jpeg")]
            image = next((p for p in images if p.exists()), None)
            assert image is not None, f"Missing page image: {images[0]}"
            with Image.open(image) as img:
                assert img.size == (width, height), (book, index, img.size, (width, height))
            texts = [obj for obj in objects.values() if obj["kind"] == "text"]
            pages.append({
                "book": book, "page_index": index, "image_path": str(image.resolve()),
                "width": width, "height": height, "texts": texts,
                "characters": [obj for obj in objects.values() if obj["kind"] == "body"],
                "faces": [obj for obj in objects.values() if obj["kind"] == "face"],
                "frames": [obj for obj in objects.values() if obj["kind"] == "frame"],
                "speaker_links": links,
                "unlinked_texts": [{"text_id": obj["id"], "speaker_status": "UNLABELED"} for obj in texts if obj["id"] not in links],
            })
    assert speaker_types["body"] > 0, "No visible body-speaker links found."
    return pages, dict(speaker_types), issues


def show_page(page, text_id=None, predictions=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.imshow(read_image(page["image_path"]))
    texts = page["texts"] if isinstance(page["texts"], dict) else {o["id"]: o for o in page["texts"]}
    characters = page["characters"] if isinstance(page["characters"], dict) else {o["id"]: o for o in page["characters"]}
    for prefix, objects, color in [("T", texts, "blue"), ("C", characters, "orange")]:
        for oid, obj in objects.items():
            x1, y1, x2, y2 = obj["bbox"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1, f"{prefix}:{oid}", color=color, fontsize=8, backgroundcolor="white")
    links = page["speaker_links"] if text_id is None else {text_id: page["speaker_links"][text_id]}
    for tid, sid in links.items():
        if tid in texts and sid in characters:
            ax.annotate("", xy=center(characters[sid]["bbox"]), xytext=center(texts[tid]["bbox"]), arrowprops={"arrowstyle": "->", "color": "green", "lw": 2})
    if predictions:
        for name, sid, color in predictions:
            ax.annotate(name, xy=center(characters[sid]["bbox"]), xytext=center(texts[text_id]["bbox"]), color=color, arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
    ax.set_title(f"{page['book']} / page {page['page_index']} — green = GT")
    ax.axis("off")
    plt.show()
    plt.close(fig)


def load_magi(device):
    # Disable OCR initialization and backbone downloads while loading all detection weights locally.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    config.disable_ocr = True
    config.detection_model_config.use_pretrained_backbone = False
    model, loading = AutoModel.from_pretrained(str(MODEL_PATH), config=config, trust_remote_code=True, local_files_only=True, output_loading_info=True)
    missing = [key for key in loading["missing_keys"] if not key.startswith("ocr_model.")]
    assert not missing, f"Local checkpoint did not load required weights: {missing}"
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def pair_inputs(page, text_id, character_ids):
    text = page["texts"][text_id]["embedding"].float()
    characters = torch.stack([page["characters"][cid]["embedding"].float() for cid in character_ids])
    token = page["t2c_token"].float()
    return torch.cat([text.expand(len(character_ids), -1), characters, token.expand(len(character_ids), -1)], dim=1)


def load_manifest(split):
    manifest = read_json(DATA / f"{split}_candidates.json")
    extraction = read_json(DATA / "extraction_manifest.json")
    assert manifest["extraction_hash"] == digest(extraction), "Rebuild candidates after changing features."
    assert manifest["candidate_hash"] == digest(manifest["pages"])
    return manifest


def load_cached(record):
    path = DATA / "cached_features" / record["cache_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["cache_sha256"], f"Cache changed: {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate(head, manifest, device, close_threshold=0.10):
    from sklearn.metrics import roc_auc_score

    head.eval()
    rows, labels, probabilities = [], [], []
    loss_sum, pair_count = 0.0, 0
    with torch.no_grad():
        for record in manifest["pages"]:
            page = load_cached(record)
            for item in record["texts"]:
                tid, ids = item["text_id"], item["candidate_ids"]
                assert ids == sorted(page["characters"]), "Evaluation must use all matched page characters."
                meta = [pair_metadata(page, tid, cid) for cid in ids]
                logits = head(pair_inputs(page, tid, ids).to(device)).flatten().cpu()
                targets = torch.tensor([m["target"] for m in meta], dtype=torch.float32)
                assert int(targets.sum()) == 1
                loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
                pair_count += len(ids)
                order = sorted(range(len(ids)), key=lambda i: (-float(logits[i]), ids[i]))
                nearest = next(m for m in meta if m["is_nearest_character"])
                truth = page["speaker_links"][tid]
                rows.append({
                    "cache_file": record["cache_file"], "book": page["book"], "page_index": page["page_index"],
                    "text_id": tid, "truth": truth, "prediction": ids[order[0]],
                    "top1": ids[order[0]] == truth, "top2": truth in [ids[i] for i in order[:2]],
                    "non_nearest": nearest["character_id"] != truth,
                    "close_wrong_nearest": nearest["character_id"] != truth and nearest["normalized_distance"] <= close_threshold,
                    "candidate_count": len(ids), "candidate_ids": ids, "scores": logits.sigmoid().tolist(),
                })
                labels.extend(targets.tolist())
                probabilities.extend(logits.sigmoid().tolist())
    assert rows, "No evaluable speaker links remain. Check matching coverage."
    subset = lambda key: [r for r in rows if r[key]]
    accuracy = lambda selected: sum(r["top1"] for r in selected) / len(selected) if selected else None
    metrics = {
        "texts": len(rows), "pairs": pair_count, "loss": loss_sum / pair_count,
        "top1": accuracy(rows), "top2": sum(r["top2"] for r in rows) / len(rows),
        "non_nearest_count": len(subset("non_nearest")), "non_nearest_top1": accuracy(subset("non_nearest")),
        "close_wrong_count": len(subset("close_wrong_nearest")), "close_wrong_top1": accuracy(subset("close_wrong_nearest")),
        "single_candidate_count": sum(r["candidate_count"] == 1 for r in rows),
        "multi_candidate_top1": accuracy([r for r in rows if r["candidate_count"] > 1]),
        "pair_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "close_threshold": close_threshold,
    }
    return metrics, rows, {"labels": labels, "probabilities": probabilities}
