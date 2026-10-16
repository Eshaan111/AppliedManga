"""Generation-based evaluation for manga transcript predictions.

Scores without evaluable support are None (JSON null), never a perfect-looking
zero. CER can exceed 1 when there are many insertions.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from pprint import pprint

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from Scripts.dataUtils import verify_image_inputs


SPECIAL_SPEAKERS = {"NARRATION", "UNKNOWN"}
GENERATION_DIAGNOSTIC_FIELDS = (
    "generated_tokens",
    "reached_eos",
    "hit_max_new_tokens",
    "suspected_repetition_loop",
)


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _f1(true_positives, false_positives, false_negatives):
    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    return _safe_div(2 * precision * recall, precision + recall)


def _clean_text(text):
    """Normalize harmless Unicode and whitespace differences."""
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.split()).strip()


def _speaker_name(speaker):
    return _clean_text(speaker).upper()


def _parse_prediction(raw_prediction):
    """Parse and validate the expected pages -> dialogues structure."""
    if not isinstance(raw_prediction, str):
        return None, False, False
    try:
        data = json.loads(raw_prediction.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False, False

    parse_valid = True
    # Training targets currently use {"sequence_id": ..., "pages": [...]},
    # while the prompt asks for the pages array. Accept both formats.
    pages = data.get("pages") if isinstance(data, dict) else data

    schema_valid = isinstance(pages, list)
    if schema_valid:
        for page in pages:
            if not isinstance(page, list):
                schema_valid = False
                break
            for dialogue in page:
                if (
                    not isinstance(dialogue, dict)
                    or set(dialogue) != {"speaker", "text"}
                    or not isinstance(dialogue.get("speaker"), str)
                    or not isinstance(dialogue.get("text"), str)
                ):
                    schema_valid = False
                    break
            if not schema_valid:
                break

    return pages if schema_valid else None, parse_valid, schema_valid


def _format_diagnostics(raw_prediction, expected_pages):
    """Explain strict validation failures without repairing outputs for scoring."""
    issues = []
    candidate = raw_prediction
    fenced = isinstance(candidate, str) and candidate.strip().startswith("```")
    if fenced:
        issues.append("Markdown code fence: return raw JSON only.")
        lines = candidate.strip().splitlines()
        # Inspect the fenced payload only to explain additional schema failures.
        candidate = "\n".join(lines[1:-1]) if lines[-1] == "```" else candidate
    try:
        data = json.loads(candidate)
        decoded = True
    except (json.JSONDecodeError, TypeError) as error:
        issues.append(f"JSON parse failed: {error}")
        data = None
        decoded = False
    pages = data.get("pages") if isinstance(data, dict) else data
    if decoded:
        if not isinstance(pages, list):
            issues.append("Expected a pages list (or an object containing 'pages').")
        elif pages and all(isinstance(page, dict) for page in pages):
            issues.append("Flat dialogue list: expected one inner list per image.")
        else:
            if len(pages) != expected_pages:
                issues.append(f"Page count mismatch: expected {expected_pages}, got {len(pages)}.")
            for page_index, page in enumerate(pages):
                if not isinstance(page, list):
                    issues.append(f"Page {page_index + 1}: expected a dialogue list.")
                    continue
                for item_index, item in enumerate(page):
                    if (not isinstance(item, dict)
                        or set(item) != {"speaker", "text"}
                        or not all(isinstance(item.get(key), str) for key in ("speaker", "text"))):
                        issues.append(f"Page {page_index + 1}, item {item_index + 1}: expected exactly string-valued speaker and text fields.")
    _, parse_valid, schema_valid = _parse_prediction(raw_prediction)
    return {
        "json_valid": parse_valid,
        "schema_valid": schema_valid,
        "page_count_matches": len(pages) == expected_pages if schema_valid else None,
        "issues": issues,
    }


def _edit_operations(reference, prediction):
    """Return Levenshtein substitution, deletion, and insertion counts."""
    rows = len(reference) + 1
    columns = len(prediction) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == prediction[column - 1]:
                distance[row][column] = distance[row - 1][column - 1]
            else:
                distance[row][column] = 1 + min(
                    distance[row - 1][column - 1],
                    distance[row - 1][column],
                    distance[row][column - 1],
                )

    substitutions = deletions = insertions = 0
    row, column = len(reference), len(prediction)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == prediction[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif (
            row
            and column
            and distance[row][column] == distance[row - 1][column - 1] + 1
        ):
            substitutions += 1
            row -= 1
            column -= 1
        elif row and distance[row][column] == distance[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return substitutions, deletions, insertions


def _edit_distance(reference, prediction):
    """Character-level Levenshtein distance."""
    return sum(_edit_operations(reference, prediction))


def _match_page(true_page, predicted_page, minimum_similarity):
    """Optimally make one-to-one dialogue matches using text similarity.

    The assignment prioritizes the number of matches above total similarity,
    then maximizes similarity among assignments with the same match count.
    """
    if not true_page or not predicted_page:
        return []

    similarities = []
    for true_index, true_dialogue in enumerate(true_page):
        true_text = _clean_text(true_dialogue["text"]).casefold()
        row = []
        for pred_index, pred_dialogue in enumerate(predicted_page):
            pred_text = _clean_text(pred_dialogue["text"]).casefold()
            row.append(SequenceMatcher(None, true_text, pred_text).ratio())
        similarities.append(row)

    # The bonus is larger than the maximum possible total similarity, making
    # match count the primary objective and similarity the tie-breaker.
    match_bonus = min(len(true_page), len(predicted_page)) + 1.0
    costs = [
        [-(match_bonus + value) if value >= minimum_similarity else 0.0 for value in row]
        for row in similarities
    ]
    rows, columns = linear_sum_assignment(costs)
    matches = [
        (int(row), int(column), similarities[row][column])
        for row, column in zip(rows, columns)
        if similarities[row][column] >= minimum_similarity
    ]
    return sorted(matches)


def _speaker_mapping(matched_items):
    """Find the best permutation from predicted character IDs to true IDs."""
    true_names = sorted(
        {
            item["true_speaker"]
            for item in matched_items
            if item["true_speaker"] not in SPECIAL_SPEAKERS
        }
    )
    pred_names = sorted(
        {
            item["pred_speaker"]
            for item in matched_items
            if item["pred_speaker"] not in SPECIAL_SPEAKERS
        }
    )
    if not true_names or not pred_names:
        return {}

    counts = [[0 for _ in true_names] for _ in pred_names]
    for item in matched_items:
        if item["pred_speaker"] in pred_names and item["true_speaker"] in true_names:
            row = pred_names.index(item["pred_speaker"])
            column = true_names.index(item["true_speaker"])
            counts[row][column] += 1

    max_count = max(max(row) for row in counts)
    costs = [[max_count - value for value in row] for row in counts]
    rows, columns = linear_sum_assignment(costs)
    return {pred_names[row]: true_names[column] for row, column in zip(rows, columns)}


def _b3_values(matched_items):
    """Return per-dialogue B-cubed precision and recall values."""
    identity_items = [
        item
        for item in matched_items
        if item["true_speaker"] not in SPECIAL_SPEAKERS
        and item["pred_speaker"] not in SPECIAL_SPEAKERS
    ]
    values = []
    for item in identity_items:
        true_cluster = {
            other["id"]
            for other in identity_items
            if other["true_speaker"] == item["true_speaker"]
        }
        pred_cluster = {
            other["id"]
            for other in identity_items
            if other["pred_speaker"] == item["pred_speaker"]
        }
        overlap = len(true_cluster & pred_cluster)
        values.append((overlap / len(pred_cluster), overlap / len(true_cluster)))
    return values


def calculate_metrics(records, minimum_similarity=0.50):
    """Calculate dataset-level metrics from raw prediction records."""
    records = list(records)
    totals = defaultdict(float)
    b3_values = []

    for sequence_number, record in enumerate(records):
        predicted_pages, parse_valid, schema_valid = _parse_prediction(record["prediction"])
        true_pages = record["target"]
        totals["sequences"] += 1
        totals["json_parse_valid"] += parse_valid
        totals["schema_valid"] += schema_valid

        if not schema_valid:
            totals["invalid_dialogue_fn"] += sum(len(page) for page in true_pages)
            missing_text = sum(
                len(_clean_text(dialogue["text"]))
                for page in true_pages
                for dialogue in page
            )
            # CER is undefined for an output that does not expose transcript
            # fields. Keep it out of valid-output CER and report a separate
            # deletion-only worst-case diagnostic for failed records.
            totals["invalid_reference_characters"] += missing_text
            continue

        totals["valid_sequences"] += 1
        totals["page_count_matches"] += len(true_pages) == len(predicted_pages)
        totals["missing_pages"] += max(0, len(true_pages) - len(predicted_pages))
        totals["extra_pages"] += max(0, len(predicted_pages) - len(true_pages))
        totals["unknown_speakers"] += sum(
            _speaker_name(item["speaker"]) == "UNKNOWN"
            for page in predicted_pages for item in page
        )

        # Missing or extra pages are treated as empty pages on the other side.
        page_count = max(len(true_pages), len(predicted_pages))
        matched_items = []
        literal_sequence_is_exact = len(true_pages) == len(predicted_pages)

        for page_index in range(page_count):
            true_page = true_pages[page_index] if page_index < len(true_pages) else []
            pred_page = predicted_pages[page_index] if page_index < len(predicted_pages) else []
            matches = _match_page(true_page, pred_page, minimum_similarity)
            matched_true = {true_index for true_index, _, _ in matches}
            matched_pred = {pred_index for _, pred_index, _ in matches}

            totals["dialogue_tp"] += len(matches)
            totals["dialogue_fn"] += len(true_page) - len(matches)
            totals["dialogue_fp"] += len(pred_page) - len(matches)

            for true_index, pred_index, _ in matches:
                true_dialogue = true_page[true_index]
                pred_dialogue = pred_page[pred_index]
                true_text = _clean_text(true_dialogue["text"])
                pred_text = _clean_text(pred_dialogue["text"])
                true_speaker = _speaker_name(true_dialogue["speaker"])
                pred_speaker = _speaker_name(pred_dialogue["speaker"])

                substitutions, deletions, insertions = _edit_operations(
                    true_text, pred_text
                )
                totals["substitutions"] += substitutions
                totals["deletions"] += deletions
                totals["insertions"] += insertions
                totals["edit_distance"] += substitutions + deletions + insertions
                totals["reference_characters"] += len(true_text)
                matched_items.append(
                    {
                        "id": (sequence_number, page_index, true_index),
                        "page": page_index,
                        "true_index": true_index,
                        "pred_index": pred_index,
                        "true_speaker": true_speaker,
                        "pred_speaker": pred_speaker,
                    }
                )

                true_is_narration = true_speaker == "NARRATION"
                pred_is_narration = pred_speaker == "NARRATION"
                totals["narration_tp"] += true_is_narration and pred_is_narration
                totals["narration_fn"] += true_is_narration and not pred_is_narration
                totals["narration_fp"] += pred_is_narration and not true_is_narration

            # Unmatched true text is a deletion; unmatched predicted text is
            # an insertion. This prevents CER from looking artificially good
            # when the model misses an entire dialogue.
            for index, dialogue in enumerate(true_page):
                if index not in matched_true:
                    text = _clean_text(dialogue["text"])
                    totals["edit_distance"] += len(text)
                    totals["deletions"] += len(text)
                    totals["reference_characters"] += len(text)
            for index, dialogue in enumerate(pred_page):
                if index not in matched_pred:
                    inserted = len(_clean_text(dialogue["text"]))
                    totals["edit_distance"] += inserted
                    totals["insertions"] += inserted

            totals["narration_fn"] += sum(
                _speaker_name(true_page[index]["speaker"]) == "NARRATION"
                for index in range(len(true_page))
                if index not in matched_true
            )
            totals["narration_fp"] += sum(
                _speaker_name(pred_page[index]["speaker"]) == "NARRATION"
                for index in range(len(pred_page))
                if index not in matched_pred
            )

            # Pairwise reading order among dialogues that were matched.
            for left in range(len(matches)):
                for right in range(left + 1, len(matches)):
                    first = matches[left]
                    second = matches[right]
                    totals["order_pairs"] += 1
                    totals["correct_order_pairs"] += first[1] < second[1]

            expected_page = [
                {
                    "speaker": _speaker_name(item["speaker"]),
                    "text": _clean_text(item["text"]),
                }
                for item in true_page
            ]
            actual_page = [
                {
                    "speaker": _speaker_name(item["speaker"]),
                    "text": _clean_text(item["text"]),
                }
                for item in pred_page
            ]
            literal_sequence_is_exact = (
                literal_sequence_is_exact and expected_page == actual_page
            )

        mapping = _speaker_mapping(matched_items)
        for item in matched_items:
            predicted = item["pred_speaker"]
            mapped_prediction = predicted if predicted in SPECIAL_SPEAKERS else mapping.get(predicted)
            totals["speaker_items"] += 1
            totals["correct_speakers"] += mapped_prediction == item["true_speaker"]

        # Count identity changes only between appearances on different pages.
        by_true_speaker = defaultdict(list)
        for item in matched_items:
            if item["true_speaker"] not in SPECIAL_SPEAKERS:
                by_true_speaker[item["true_speaker"]].append(item)
        for appearances in by_true_speaker.values():
            appearances.sort(key=lambda item: (item["page"], item["true_index"]))
            for previous, current in zip(appearances, appearances[1:]):
                if previous["page"] != current["page"]:
                    totals["identity_transitions"] += 1
                    totals["identity_switches"] += (
                        previous["pred_speaker"] != current["pred_speaker"]
                    )

        b3_values.extend(_b3_values(matched_items))

        # Character IDs are arbitrary. Exact match must use the same optimal
        # speaker permutation as speaker accuracy, otherwise a perfectly
        # equivalent character1/character2 swap incorrectly scores zero.
        mapped_sequence_is_exact = len(true_pages) == len(predicted_pages)
        if mapped_sequence_is_exact:
            for true_page, pred_page in zip(true_pages, predicted_pages):
                if len(true_page) != len(pred_page):
                    mapped_sequence_is_exact = False
                    break
                for true_dialogue, pred_dialogue in zip(true_page, pred_page):
                    predicted_speaker = _speaker_name(pred_dialogue["speaker"])
                    mapped_speaker = (
                        predicted_speaker
                        if predicted_speaker in SPECIAL_SPEAKERS
                        else mapping.get(predicted_speaker)
                    )
                    if (
                        _clean_text(true_dialogue["text"])
                        != _clean_text(pred_dialogue["text"])
                        or mapped_speaker != _speaker_name(true_dialogue["speaker"])
                    ):
                        mapped_sequence_is_exact = False
                        break
                if not mapped_sequence_is_exact:
                    break

        totals["literal_exact_sequences"] += literal_sequence_is_exact
        totals["exact_sequences"] += mapped_sequence_is_exact

    b3_precision = (
        sum(precision for precision, _ in b3_values) / len(b3_values)
        if b3_values
        else 0.0
    )
    b3_recall = (
        sum(recall for _, recall in b3_values) / len(b3_values)
        if b3_values
        else 0.0
    )

    cer_valid = _safe_div(totals["edit_distance"], totals["reference_characters"])
    cer_including_failures = _safe_div(
        totals["edit_distance"] + totals["invalid_reference_characters"],
        totals["reference_characters"] + totals["invalid_reference_characters"],
    )
    dialogue_text_match_f1 = _f1(
        totals["dialogue_tp"], totals["dialogue_fp"], totals["dialogue_fn"]
    )
    matched_order_accuracy = _safe_div(
        totals["correct_order_pairs"], totals["order_pairs"]
    )
    matched_speaker_accuracy = _safe_div(
        totals["correct_speakers"], totals["speaker_items"]
    )

    metrics = {
        # CER is calculated only where transcript fields can be parsed. The
        # parse/schema rates must always be read alongside it.
        "cer": cer_valid,
        "cer_valid_outputs": cer_valid,
        "cer_evaluated_sequences": int(totals["valid_sequences"]),
        "cer_including_failures_as_deletions": cer_including_failures,
        "cer_substitution_rate": _safe_div(
            totals["substitutions"], totals["reference_characters"]
        ),
        "cer_deletion_rate": _safe_div(
            totals["deletions"], totals["reference_characters"]
        ),
        "cer_insertion_rate": _safe_div(
            totals["insertions"], totals["reference_characters"]
        ),
        "dialogue_text_match_f1": dialogue_text_match_f1,
        "dialogue_f1": dialogue_text_match_f1,
        "matched_dialogue_order_accuracy": matched_order_accuracy,
        "reading_order_accuracy": matched_order_accuracy,
        "matched_speaker_accuracy": matched_speaker_accuracy,
        "speaker_accuracy": matched_speaker_accuracy,
        "b3_f1": _safe_div(2 * b3_precision * b3_recall, b3_precision + b3_recall),
        "identity_switch_rate": _safe_div(
            totals["identity_switches"], totals["identity_transitions"]
        ),
        "narration_f1": _f1(
            totals["narration_tp"], totals["narration_fp"], totals["narration_fn"]
        ),
        "gt_narration_count": int(
            totals["narration_tp"] + totals["narration_fn"]
        ),
        "predicted_narration_count": int(
            totals["narration_tp"] + totals["narration_fp"]
        ),
        "matched_narration_count": int(totals["narration_tp"]),
        "json_parse_rate": _safe_div(totals["json_parse_valid"], totals["sequences"]),
        "schema_compliance_rate": _safe_div(totals["schema_valid"], totals["sequences"]),
        "sequence_exact_match": _safe_div(totals["exact_sequences"], totals["sequences"]),
        "literal_sequence_exact_match": _safe_div(
            totals["literal_exact_sequences"], totals["sequences"]
        ),
        "evaluated_sequences": int(totals["sequences"]),
    }
    # Keep end-to-end failure penalties explicitly separate from conditional
    # content scores. An invalid response reveals nothing about transcription.
    metrics["dialogue_f1_including_format_failures"] = _f1(
        totals["dialogue_tp"], totals["dialogue_fp"],
        totals["dialogue_fn"] + totals["invalid_dialogue_fn"],
    ) if totals["sequences"] else None
    support = {
        "valid_outputs": int(totals["valid_sequences"]),
        "total_outputs": int(totals["sequences"]),
        "reference_characters": int(totals["reference_characters"]),
        "matched_dialogues": int(totals["dialogue_tp"]),
        "missing_dialogues": int(totals["dialogue_fn"]),
        "extra_dialogues": int(totals["dialogue_fp"]),
        "unscored_target_dialogues": int(totals["invalid_dialogue_fn"]),
        "speaker_items": int(totals["speaker_items"]),
        "correct_speakers": int(totals["correct_speakers"]),
        "unknown_speakers": int(totals["unknown_speakers"]),
        "order_pairs": int(totals["order_pairs"]),
        "correct_order_pairs": int(totals["correct_order_pairs"]),
        "identity_transitions": int(totals["identity_transitions"]),
        "identity_switches": int(totals["identity_switches"]),
        "identity_items": len(b3_values),
        "missing_pages": int(totals["missing_pages"]),
        "extra_pages": int(totals["extra_pages"]),
        "substitutions": int(totals["substitutions"]),
        "deletions": int(totals["deletions"]),
        "insertions": int(totals["insertions"]),
    }
    metrics["support"] = support
    metrics["dialogue_precision"] = (
        _safe_div(totals["dialogue_tp"], totals["dialogue_tp"] + totals["dialogue_fp"])
        if totals["dialogue_tp"] + totals["dialogue_fp"] else None
    )
    metrics["dialogue_recall"] = (
        _safe_div(totals["dialogue_tp"], totals["dialogue_tp"] + totals["dialogue_fn"])
        if totals["dialogue_tp"] + totals["dialogue_fn"] else None
    )
    metrics["page_count_match_rate"] = (
        totals["page_count_matches"] / totals["valid_sequences"]
        if totals["valid_sequences"] else None
    )
    unavailable = {}
    def mark_unavailable(keys, reason):
        for key in keys:
            metrics[key] = None
            unavailable[key] = reason

    if not totals["reference_characters"]:
        mark_unavailable(
            ("cer", "cer_valid_outputs", "cer_substitution_rate", "cer_deletion_rate", "cer_insertion_rate"),
            "invalid output" if totals["sequences"] and not totals["valid_sequences"] else "no reference characters",
        )
    if not totals["valid_sequences"]:
        mark_unavailable(("dialogue_f1", "dialogue_text_match_f1", "dialogue_precision", "dialogue_recall", "page_count_match_rate", "narration_f1"), "invalid output" if totals["sequences"] else "no samples")
    elif not totals["dialogue_tp"] + totals["dialogue_fp"] + totals["dialogue_fn"]:
        mark_unavailable(("dialogue_f1", "dialogue_text_match_f1"), "no dialogues")
    if not totals["speaker_items"]:
        mark_unavailable(("matched_speaker_accuracy", "speaker_accuracy"), "invalid output" if totals["sequences"] and not totals["valid_sequences"] else "no matched dialogues")
    if not totals["order_pairs"]:
        mark_unavailable(("matched_dialogue_order_accuracy", "reading_order_accuracy"), "no matched dialogue pairs")
    if not totals["identity_transitions"]:
        mark_unavailable(("identity_switch_rate",), "no cross-page identity transitions")
    if not b3_values:
        mark_unavailable(("b3_f1",), "no matched character identities")
    if totals["valid_sequences"] and not totals["narration_tp"] + totals["narration_fp"] + totals["narration_fn"]:
        mark_unavailable(("narration_f1",), "no target or predicted narration")
    if not totals["reference_characters"] + totals["invalid_reference_characters"]:
        mark_unavailable(("cer_including_failures_as_deletions",), "no reference characters")
    if not totals["sequences"]:
        mark_unavailable(("json_parse_rate", "schema_compliance_rate", "sequence_exact_match", "literal_sequence_exact_match"), "no samples")
    metrics["unavailable_reasons"] = unavailable
    formats = [_format_diagnostics(record["prediction"], len(record["target"])) for record in records]
    metrics["format_diagnostics"] = {
        "json_failures": sum(not item["json_valid"] for item in formats),
        "schema_failures_after_json_parse": sum(item["json_valid"] and not item["schema_valid"] for item in formats),
        "page_count_mismatches": sum(item["page_count_matches"] is False for item in formats),
    }
    return metrics


def metric_summary_rows(metrics):
    """Small presentation table shared by notebooks and console inspection."""
    support = metrics["support"]
    rows = []
    for label, key, explanation in (
        ("Format valid?", "schema_compliance_rate", f"{support['valid_outputs']}/{support['total_outputs']} outputs usable; check page-count diagnostics"),
        ("Text error rate (lower is better)", "cer_valid_outputs", f"{support['reference_characters']} reference characters on valid outputs"),
        ("Dialogue F1", "dialogue_text_match_f1", f"{support['matched_dialogues']} matched, {support['missing_dialogues']} missing, {support['extra_dialogues']} extra; {support['unscored_target_dialogues']} target lines unscored due to format failures"),
        ("Speaker accuracy", "matched_speaker_accuracy", f"{support['correct_speakers']}/{support['speaker_items']} matched dialogues correct after character-ID alignment"),
    ):
        value = metrics[key]
        result = f"N/A — {metrics['unavailable_reasons'].get(key, 'no support')}" if value is None else f"{value:.3f}"
        if key == "schema_compliance_rate" and support["total_outputs"] == 1:
            result = "Yes" if value == 1 else "No"
        rows.append({"Result": label, "Value": result, "What it tells us": explanation})
    return rows


def print_metric_summary(metrics):
    for row in metric_summary_rows(metrics):
        print(f"{row['Result']}: {row['Value']} ({row['What it tells us']})")


def display_metric_summary(metrics, report=None):
    """Notebook summary with optional, collapsed diagnostics below it."""
    import html
    import pandas as pd
    from IPython.display import HTML, display

    display(pd.DataFrame(metric_summary_rows(metrics)))
    diagnostics = {"metrics": metrics}
    if report is not None:
        diagnostics.update({key: report[key] for key in
                            ("format_diagnostics", "page_diagnostics", "generation_diagnostics")})
        diagnostics["image_input_diagnostics"] = report.get("image_input_diagnostics")
        for issue in report["format_diagnostics"]["issues"]:
            print(f"Format: {issue}")
    rendered = html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    display(HTML(f"<details><summary>Detailed scores, support counts and diagnostics (null = N/A)</summary><pre>{rendered}</pre></details>"))


def _has_repeated_token_pattern(token_ids, repeat_threshold, max_pattern_width=8):
    """Detect a short token pattern repeated consecutively many times."""
    if repeat_threshold <= 1:
        return bool(token_ids)
    for width in range(1, max_pattern_width + 1):
        required = width * repeat_threshold
        if len(token_ids) < required:
            continue
        for start in range(len(token_ids) - required + 1):
            pattern = token_ids[start : start + width]
            if all(
                token_ids[offset : offset + width] == pattern
                for offset in range(start + width, start + required, width)
            ):
                return True
    return False


def _generation_diagnostics(records):
    token_counts = [record["generated_tokens"] for record in records]
    return {
        "generated_sequences": len(records),
        "hit_max_new_tokens": sum(
            record["hit_max_new_tokens"] for record in records
        ),
        "suspected_repetition_loops": sum(
            record["suspected_repetition_loop"] for record in records
        ),
        "min_generated_tokens": min(token_counts, default=0),
        "max_generated_tokens": max(token_counts, default=0),
        "mean_generated_tokens": _safe_div(sum(token_counts), len(token_counts)),
    }


def load_prediction_records(predictions_file):
    """Load prediction records written by :func:`evaluate_model`."""
    predictions_path = Path(predictions_file)
    with predictions_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _sequence_report(record, minimum_similarity):
    raw_prediction = record["prediction"]
    try:
        parsed_prediction = json.loads(raw_prediction)
        prediction_is_valid_json = True
    except (json.JSONDecodeError, TypeError):
        parsed_prediction = None
        prediction_is_valid_json = False

    predicted_pages, _, schema_valid = _parse_prediction(raw_prediction)
    format_diagnostics = _format_diagnostics(raw_prediction, len(record["target"]))
    page_diagnostics = []
    if schema_valid:
        for page_index in range(max(len(record["target"]), len(predicted_pages))):
            target = record["target"][page_index] if page_index < len(record["target"]) else []
            prediction = predicted_pages[page_index] if page_index < len(predicted_pages) else []
            matches = _match_page(target, prediction, minimum_similarity)
            matched_true = {i for i, _, _ in matches}
            matched_pred = {j for _, j, _ in matches}
            page_diagnostics.append({
                "page": page_index + 1,
                "target_dialogues": len(target),
                "predicted_dialogues": len(prediction),
                "matches": [{"target_index": i, "prediction_index": j, "text_similarity": similarity,
                             "target_text": target[i]["text"], "prediction_text": prediction[j]["text"],
                             "target_speaker": target[i]["speaker"], "prediction_speaker": prediction[j]["speaker"]}
                            for i, j, similarity in matches],
                "unmatched_target": [{"index": i, **item} for i, item in enumerate(target) if i not in matched_true],
                "unmatched_prediction": [{"index": j, **item} for j, item in enumerate(prediction) if j not in matched_pred],
                "order_errors": sum(first[1] > second[1] for n, first in enumerate(matches) for second in matches[n + 1:]),
            })
    metrics = calculate_metrics([record], minimum_similarity=minimum_similarity)
    return {
        "sequence_id": record["sequence_id"],
        "target": record["target"],
        "prediction": parsed_prediction,
        "prediction_raw": raw_prediction,
        "prediction_is_valid_json": prediction_is_valid_json,
        "metrics": metrics,
        "summary": metric_summary_rows(metrics),
        "format_diagnostics": format_diagnostics,
        "page_diagnostics": page_diagnostics,
        "minimum_text_similarity": minimum_similarity,
        "image_input_diagnostics": record.get("image_input_diagnostics"),
        "generation_diagnostics": {
            field: record.get(field) for field in GENERATION_DIAGNOSTIC_FIELDS
        },
    }


def prediction_report(record, minimum_similarity=0.50):
    """Build an inspectable report directly from a target/raw prediction pair."""
    return _sequence_report(record, minimum_similarity)


def inspect_prediction(
    predictions_file,
    *,
    sequence_id=None,
    index=0,
    minimum_similarity=0.50,
    print_output=True,
):
    """Inspect one saved target/prediction pair and its sequence metrics.

    Select a record with ``sequence_id`` or, when it is omitted, its zero-based
    ``index``. The structured report is returned even when printing is enabled.
    """
    records = load_prediction_records(predictions_file)
    if not records:
        raise ValueError(f"No prediction records found in {predictions_file}")

    if sequence_id is not None:
        record = next(
            (
                item
                for item in records
                if item.get("sequence_id") == sequence_id
            ),
            None,
        )
        if record is None:
            available = [item.get("sequence_id") for item in records]
            raise KeyError(
                f"Unknown sequence_id {sequence_id!r}. Available IDs: {available}"
            )
    else:
        try:
            record = records[index]
        except IndexError as error:
            raise IndexError(
                f"Prediction index {index} is outside 0..{len(records) - 1}"
            ) from error

    report = _sequence_report(record, minimum_similarity)
    if print_output:
        print(f"SEQUENCE\n{report['sequence_id']}")
        print("\nTARGET")
        print(json.dumps(report["target"], ensure_ascii=False, indent=2))
        print("\nPREDICTION")
        if report["prediction_is_valid_json"]:
            print(
                json.dumps(
                    report["prediction"], ensure_ascii=False, indent=2
                )
            )
        else:
            print("INVALID JSON:")
            print(report["prediction_raw"])
        print("\nRESULT")
        print_metric_summary(report["metrics"])
        print("\nFORMAT DIAGNOSTICS")
        for issue in report["format_diagnostics"]["issues"]:
            print(f"- {issue}")
        if not report["format_diagnostics"]["issues"]:
            print("Raw JSON and page/item structure are valid.")
        print("\nDETAILED DIAGNOSTICS (content counts use valid outputs only)")
        pprint(report["metrics"], sort_dicts=False)
        pprint(report["page_diagnostics"], sort_dicts=False)
        print("\nGENERATION DIAGNOSTICS")
        pprint(report["generation_diagnostics"], sort_dicts=False)
        print("\nIMAGE INPUT DIAGNOSTICS")
        pprint(report["image_input_diagnostics"], sort_dicts=False)

    return report


def save_per_sequence_reports(
    predictions_file,
    output_file=None,
    *,
    minimum_similarity=0.50,
):
    """Calculate and save one target/prediction/metrics report per sequence."""
    predictions_path = Path(predictions_file)
    records = load_prediction_records(predictions_path)
    reports = [
        _sequence_report(record, minimum_similarity) for record in records
    ]
    output_path = (
        Path(output_file)
        if output_file is not None
        else predictions_path.with_name("per_sequence_results.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for report in reports:
            file.write(json.dumps(report, ensure_ascii=False) + "\n")
    return output_path


def evaluate_model(
    model,
    processor,
    dataset,
    collator,
    batch_size=1,
    max_new_tokens=1024,
    minimum_similarity=0.50,
    repetition_penalty=1.05,
    repetition_run_threshold=20,
    output_dir=None,
    prediction_formatter=None,
    generation_only=False,
):
    """Generate predictions; generation_only saves raw outputs without JSON scoring."""
    if generation_only and prediction_formatter is not None:
        raise ValueError('Generation-only evaluation cannot load a prediction formatter')
    if generation_only and output_dir is None:
        raise ValueError('Generation-only evaluation requires an output directory')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    device = next(model.parameters()).device
    records = []
    was_training = model.training
    model.eval()

    for batch_number, (inputs, metadata) in enumerate(loader, start=1):
        image_audits = verify_image_inputs(processor, inputs, metadata)
        inputs = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=repetition_penalty,
            )

        prompt_width = inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_width:]
        predictions = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        eos_token_id = getattr(model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        eos_ids.discard(None)
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)

        generation_info = []
        for token_row in generated_only:
            token_ids = token_row.tolist()
            eos_positions = [
                index for index, token_id in enumerate(token_ids) if token_id in eos_ids
            ]
            reached_eos = bool(eos_positions)
            if reached_eos:
                generated_tokens = eos_positions[0] + 1
            elif pad_token_id is not None and pad_token_id in token_ids:
                generated_tokens = token_ids.index(pad_token_id)
            else:
                generated_tokens = len(token_ids)

            active_ids = token_ids[:generated_tokens]
            generation_info.append(
                {
                    "generated_tokens": generated_tokens,
                    "reached_eos": reached_eos,
                    "hit_max_new_tokens": (
                        not reached_eos and generated_tokens >= max_new_tokens
                    ),
                    "suspected_repetition_loop": (
                        _has_repeated_token_pattern(
                            active_ids, repetition_run_threshold
                        )
                    ),
                }
            )

        for prediction, item, info, audit in zip(predictions, metadata, generation_info, image_audits):
            formatter_fields = {}
            if prediction_formatter is not None:
                formatter_fields['raw_transcript'] = prediction
                try:
                    prediction = prediction_formatter(prediction)
                    formatter_fields['formatter_error'] = None
                except ValueError as error:
                    formatter_fields['formatter_error'] = str(error)
                    # Invalid JSON keeps formatting failures in failure-aware CER.
                    prediction = ''
                formatter_fields['raw_formatter_prediction'] = getattr(prediction_formatter, 'last_raw_output', None)
            records.append(
                {
                    "sequence_id": item["sequence_id"],
                    "target": item["target"],
                    "prediction": prediction,
                    "image_paths": item["image_paths"],
                    "image_input_diagnostics": audit,
                    **info,
                    **formatter_fields,
                }
            )
        print(f"Evaluated {len(records)}/{len(dataset)} sequences")

    if was_training:
        model.train()

    if generation_only:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / 'predictions.jsonl').open('w', encoding='utf-8') as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
        return None

    metrics = calculate_metrics(records, minimum_similarity=minimum_similarity)
    metrics["generation_diagnostics"] = _generation_diagnostics(records)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        save_per_sequence_reports(
            output_path / "predictions.jsonl",
            output_path / "per_sequence_results.jsonl",
            minimum_similarity=minimum_similarity,
        )

    return metrics
