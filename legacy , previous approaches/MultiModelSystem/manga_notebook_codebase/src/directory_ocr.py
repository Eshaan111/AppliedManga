"""Run the existing geometry-based OCR joiner on a standalone labelled directory."""
import csv
import hashlib
import json
from pathlib import Path

from data_utils import IMAGE_EXTS, load_jsonl, save_jsonl
from dataset_utils import save_json
from eval_utils import validate_prediction_schema
from ocr_joining import compare_groups_to_targets, join_ocr_candidates
from ocr_reports import write_page_report, write_report_index
from ocr_utils import ocr_page
from text_exclusions import filter_confirmed_sound_effects


def inspect_directory(input_dir):
    """Validate labels and three lexically ordered images per sequence, without a split."""
    input_dir = Path(input_dir).resolve()
    labels = load_jsonl(input_dir / "labels.jsonl")
    if not labels or not all(validate_prediction_schema(row) for row in labels):
        raise ValueError("labels.jsonl must contain nonempty, valid three-page sequence records")
    ids = [row["sequence_id"] for row in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sequence_id in labels.jsonl")
    image_root = input_dir / "images"
    folders = {p.name for p in image_root.iterdir() if p.is_dir()}
    if folders != set(ids):
        raise ValueError(f"Images/labels disagree: missing folders={set(ids) - folders}, "
                         f"unlabelled folders={folders - set(ids)}")
    rows = []
    for label in labels:
        sid = label["sequence_id"]
        folder = (image_root / sid).resolve()
        if not folder.is_relative_to(image_root.resolve()):
            raise ValueError(f"Unsafe sequence folder: {sid}")
        paths = sorted(p.resolve() for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if len(paths) != 3 or any(not p.is_relative_to(image_root.resolve()) for p in paths):
            raise ValueError(f"{sid}: expected exactly three safe image files, found {len(paths)}")
        # Absolute paths let the existing report writer open this standalone dataset.
        rows.append(dict(sequence_id=sid, images=[str(p) for p in paths]))
    return rows, {row["sequence_id"]: row for row in labels}


def _source_manifest(input_dir, rows):
    images = []
    for row in rows:
        for path in row["images"]:
            path = Path(path)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()  
            images.append(dict(sequence_id=row["sequence_id"],
                               image=str(path.relative_to(input_dir)), sha256=digest))
    return dict(input_dir=str(input_dir), images=images,
                ocr_settings=dict(lang="en", device="gpu:0", enable_mkldnn=False,
                                  use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=False))


def load_directory_candidates(path, rows):
    """Validate cached page evidence and select covered labelled sequences in input order."""
    cached = {}
    for row in load_jsonl(path):
        sid = row["sequence_id"]
        if sid in cached or not isinstance(row.get("pages"), list) or len(row["pages"]) != 3:
            raise ValueError(f"Invalid or duplicate OCR cache sequence: {sid}")
        for page in row["pages"]:
            if not isinstance(page, list):
                raise ValueError(f"Invalid OCR page in {sid}")
            ids = set()
            for candidate in page:
                if (not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str)
                        or candidate["id"] in ids or not isinstance(candidate.get("text"), str)
                        or not isinstance(candidate.get("box"), list) or len(candidate["box"]) != 4
                        or any(not isinstance(v, (int, float)) for v in candidate["box"])):
                    raise ValueError(f"Invalid OCR candidate in {sid}")
                ids.add(candidate["id"])
        cached[sid] = row
    selected = [row for row in rows if row["sequence_id"] in cached]
    if not selected:
        raise ValueError("Candidates contain no sequences matching this input's labels")
    return cached, selected


def run_directory(input_dir, output, *, limit=None, reviews_path=None, ocr_factory=None,
                  candidates_path=None):
    """Cache OCR for this input only, join fragments, then compare against labels.

    Completed sequences are reused after verifying source image hashes. Labels never
    enter OCR or joining. With candidates_path, only covered labelled sequences are
    joined and OCR is never run. ocr_factory supports offline integration tests.
    """
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    rows, references = inspect_directory(input_dir)
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if output == input_dir or output.is_relative_to(input_dir / "images"):
        raise ValueError("Choose an output folder separate from the input images and labels")
    source = _source_manifest(input_dir, rows)
    external_candidates = None
    if candidates_path is not None:
        candidates_path = Path(candidates_path).resolve()
        if candidates_path.is_relative_to(output):
            raise ValueError("Keep the source candidates file outside the joining output folder")
        external_candidates, covered = load_directory_candidates(candidates_path, rows)
        with candidates_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        source["candidates"] = dict(path=str(candidates_path), sha256=digest)
    source_path, cache_path = output / "source.json", output / "candidates.jsonl"
    if source_path.exists():
        if json.loads(source_path.read_text(encoding="utf-8")) != source:
            raise ValueError("Output belongs to different or changed images/candidates. Choose a new output folder.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is not an empty directory or a verified directory OCR run")
    eligible = covered if external_candidates is not None else rows
    selected = eligible[:limit] if limit is not None else eligible
    cached = {}
    if external_candidates is not None:
        cached = external_candidates
    elif cache_path.exists():
        for row in load_jsonl(cache_path):
            sid = row["sequence_id"]
            if sid in cached or sid not in references or len(row["pages"]) != 3:
                raise ValueError(f"Invalid or duplicate OCR cache sequence: {sid}")
            cached[sid] = row
    output.mkdir(parents=True, exist_ok=True)
    save_json(source, source_path)
    missing = [row for row in selected if row["sequence_id"] not in cached]
    if missing and candidates_path is None:
        if ocr_factory is None:
            from paddleocr import PaddleOCR
            ocr_factory = lambda: PaddleOCR(**source["ocr_settings"])
        ocr = ocr_factory()
        for index, row in enumerate(missing, 1):
            sid = row["sequence_id"]
            print(f"OCR {index}/{len(missing)}: {sid}", flush=True)
            cached[sid] = dict(sequence_id=sid, pages=[ocr_page(ocr, path) for path in row["images"]])
            save_jsonl(list(cached.values()), cache_path)

    reviews = {}
    if reviews_path is not None:
        for review in load_jsonl(reviews_path):
            key = (review["sequence_id"], review["page_index"])
            if key[0] not in references or key[1] not in range(3):
                raise ValueError(f"Review refers to an unknown input page: {key}")
            reviews.setdefault(key, []).append(review)
    joined_rows, reports, table = [], [], []
    for row in selected:
        sid = row["sequence_id"]
        pages = []
        for page_index, fragments in enumerate(cached[sid]["pages"]):
            kept, excluded = filter_confirmed_sound_effects(fragments, reviews.get((sid, page_index), []))
            groups = join_ocr_candidates(kept)
            pages.append(groups)
            targets = references[sid]["pages"][page_index]
            report = dict(sequence_id=sid, page_index=page_index, raw_count=len(fragments),
                          target_count=len(targets), excluded_sound_effects=excluded,
                          **compare_groups_to_targets(groups, targets))
            reports.append(report)
            for group in report["groups"]:
                table.append(dict(sequence_id=sid, page_index=page_index,
                    status="matched" if group["target_index"] is not None else "unmatched_group",
                    group_id=group["id"], candidate_ids=" ".join(group["candidate_ids"]),
                    joined_text=group["text"], target_index=group["target_index"],
                    target_text=group["target_text"], similarity=group["similarity"]))
            for target in report["unmatched_targets"]:
                table.append(dict(sequence_id=sid, page_index=page_index, status="unmatched_target",
                    group_id=None, candidate_ids=None, joined_text=None,
                    target_index=target["target_index"], target_text=target["text"], similarity=None))
        joined_rows.append(dict(sequence_id=sid, pages=pages))
    save_jsonl(joined_rows, output / "joined.jsonl")
    save_jsonl(reports, output / "allocations.jsonl")
    report_by_page = {(r["sequence_id"], r["page_index"]): r for r in reports}
    entries = []
    for sample_index, row in enumerate(selected):
        sid = row["sequence_id"]
        for page_index, fragments in enumerate(cached[sid]["pages"]):
            entries.append(write_page_report(input_dir, row, sample_index,
                report_by_page[sid, page_index], fragments, output))
    write_report_index(entries, output)
    (output / "transcripts.txt").write_text("\n\n".join(
        f"{r['sequence_id']} / page {r['page_index']}\n" +
        "\n".join(g["text"] for g in r["groups"]) for r in reports) + "\n", encoding="utf-8")
    fields = ["sequence_id", "page_index", "status", "group_id", "candidate_ids",
              "joined_text", "target_index", "target_text", "similarity"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    matches = [g for r in reports for g in r["groups"] if g["target_index"] is not None]
    groups = sum(len(r["groups"]) for r in reports)
    targets = sum(r["target_count"] for r in reports)
    summary = dict(input_dir=str(input_dir), output_dir=str(output),
        available_sequences=len(rows), sequences=len(selected), pages=len(reports),
        candidates_path=str(candidates_path) if candidates_path is not None else None,
        uncached_sequences=len(rows) - len(eligible),
        raw_fragments=sum(r["raw_count"] for r in reports), proposed_groups=groups,
        excluded_sound_effect_fragments=sum(len(r["excluded_sound_effects"]) for r in reports),
        target_items=targets, matched_targets=len(matches), unmatched_groups=groups - len(matches),
        unmatched_targets=targets - len(matches),
        high_similarity_matches=sum(g["similarity"] >= .9 for g in matches),
        mean_matched_similarity=sum(g["similarity"] for g in matches) / len(matches) if matches else None,
        matching_threshold=.5,
        note="Geometry-based OCR groups in spatial order. Labels are comparison only; "
             "bubble boundaries, speakers, and panel reading order are not verified.")
    save_json(summary, output / "summary.json")
    return summary
