"""Diagnostics; the unchanged TaskData scorer remains authoritative."""
import importlib.util
import unicodedata
from pathlib import Path

_spec = importlib.util.spec_from_file_location("manga_official_score", Path(__file__).resolve().parents[1] / "TaskData/score.py")
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)
normalize_text = official.norm
levenshtein = official.distance


def validate_prediction_schema(row):
    if not isinstance(row, dict) or set(row) != {"sequence_id", "pages"}:
        return False
    sid = row["sequence_id"]
    if not isinstance(sid, str) or not sid.strip() or sid != sid.strip():
        return False
    if not isinstance(row["pages"], list) or len(row["pages"]) != 3:
        return False
    for page in row["pages"]:
        if not isinstance(page, list):
            return False
        for item in page:
            if not isinstance(item, dict) or set(item) != {"speaker", "text"}:
                return False
            speaker, text = item["speaker"], item["text"]
            if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
                return False
            if not isinstance(text, str) or not normalize_text(text):
                return False
            if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
                return False
    return True


def cer(reference, prediction):
    a, b = normalize_text(reference), normalize_text(prediction)
    return levenshtein(a, b) / max(len(a), 1)


def page_text(page):
    return "\n".join(x["text"] for x in page)


def sequence_text_cer(reference, prediction):
    return sum(cer(page_text(a), page_text(b)) for a, b in zip(reference["pages"], prediction["pages"])) / 3


def pairwise_order_accuracy(reference_ids, predicted_ids):
    if len(set(predicted_ids)) != len(predicted_ids) or set(reference_ids) != set(predicted_ids):
        raise ValueError("Order output must be a complete permutation of candidate IDs")
    positions = {x: i for i, x in enumerate(predicted_ids)}
    pairs = [(a, b) for i, a in enumerate(reference_ids) for b in reference_ids[i + 1:]]
    return sum(positions[a] < positions[b] for a, b in pairs) / len(pairs) if pairs else 1.0


def best_speaker_mapping(reference_speakers, predicted_speakers):
    if len(reference_speakers) != len(predicted_speakers):
        raise ValueError("Speaker arrays must have equal lengths")
    return official.best_mapping([(r, p, 1) for r, p in zip(reference_speakers, predicted_speakers)
                                  if r not in {None, "UNKNOWN"}])


def speaker_accuracy(reference_speakers, predicted_speakers):
    mapping = best_speaker_mapping(reference_speakers, predicted_speakers)
    eligible = [(r, p) for r, p in zip(reference_speakers, predicted_speakers) if r not in {None, "UNKNOWN"}]
    return sum(mapping.get(p) == r for r, p in eligible) / len(eligible) if eligible else None


def binary_metrics(target, prediction):
    if len(target) != len(prediction):
        raise ValueError("Unequal diagnostic arrays")
    tp = sum(bool(a) and bool(b) for a, b in zip(target, prediction))
    fp = sum(not a and bool(b) for a, b in zip(target, prediction))
    fn = sum(bool(a) and not b for a, b in zip(target, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                accuracy=sum(bool(a) == bool(b) for a, b in zip(target, prediction)) / len(target) if target else None,
                count=len(target))


def narration_metrics(reference_speakers, predicted_speakers):
    if len(reference_speakers) != len(predicted_speakers):
        raise ValueError("Unequal speaker arrays")
    eligible = [(r, p) for r, p in zip(reference_speakers, predicted_speakers) if r not in {None, "UNKNOWN"}]
    return binary_metrics([r == "NARRATION" for r, _ in eligible], [p == "NARRATION" for _, p in eligible])


def matched_lines(ref_page, pred_page, threshold=0.8):
    weights = []
    for i, a in enumerate(ref_page):
        for j, b in enumerate(pred_page):
            left, right = normalize_text(a["text"]), normalize_text(b["text"])
            similarity = 1 - levenshtein(left, right) / max(len(left), len(right), 1)
            if similarity >= threshold:
                weights.append((str(i), str(j), similarity))
    mapping = official.best_mapping(weights)
    return [(int(i), int(j)) for j, i in mapping.items() if j != "NARRATION"]


def identity_switch_metrics(matches):
    # Switch between dominant labels on consecutive page appearances of a gold character.
    from collections import Counter
    by_character = {}
    for page, ref, pred in matches:
        if ref not in {None, "UNKNOWN", "NARRATION"}:
            by_character.setdefault(ref, {}).setdefault(page, []).append(pred)
    switches = transitions = 0
    for pages in by_character.values():
        labels = [Counter(pages[p]).most_common(1)[0][0] for p in sorted(pages)]
        switches += sum(a != b for a, b in zip(labels, labels[1:]))
        transitions += max(0, len(labels) - 1)
    return dict(identity_switch_rate=switches / transitions if transitions else None,
                switches=switches, transitions=transitions)


def empty_page_metrics(references, predictions):
    by_id = {x["sequence_id"]: x for x in predictions}
    empty = correct = extra = missing = 0
    for ref in references:
        pred = by_id.get(ref["sequence_id"])
        for i, page in enumerate(ref["pages"]):
            if not page:
                empty += 1
                if pred is None:
                    missing += 1
                else:
                    correct += not pred["pages"][i]
                    extra += len(pred["pages"][i])
    return dict(empty_pages=empty, empty_page_accuracy=correct / empty if empty else None,
                false_positive_lines=extra, missing_empty_pages=missing)
