from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from Scripts.evaluator import (
    _edit_operations,
    _match_page,
    calculate_metrics,
    inspect_prediction,
    metric_summary_rows,
    prediction_report,
    save_per_sequence_reports,
)
from tests.common import LABELS, ROOT, read_jsonl


class EvaluatorRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual_target = read_jsonl(LABELS)[0]["pages"]

    def test_perfect_actual_target_scores_as_perfect(self):
        metrics = calculate_metrics(
            [
                {
                    "target": self.actual_target,
                    "prediction": json.dumps(self.actual_target, ensure_ascii=False),
                }
            ]
        )
        self.assertEqual(metrics["cer_valid_outputs"], 0.0)
        self.assertEqual(metrics["dialogue_text_match_f1"], 1.0)
        self.assertEqual(metrics["matched_speaker_accuracy"], 1.0)
        self.assertEqual(metrics["sequence_exact_match"], 1.0)
        self.assertEqual(metrics["json_parse_rate"], 1.0)

    def test_exact_match_allows_arbitrary_character_label_permutation(self):
        prediction = copy.deepcopy(self.actual_target)
        character_names = sorted(
            {
                dialogue["speaker"]
                for page in prediction
                for dialogue in page
                if dialogue["speaker"].startswith("character")
            }
        )
        reversed_names = dict(zip(character_names, reversed(character_names)))
        self.assertTrue(any(name != reversed_names[name] for name in character_names))
        for page in prediction:
            for dialogue in page:
                dialogue["speaker"] = reversed_names.get(
                    dialogue["speaker"], dialogue["speaker"]
                )

        metrics = calculate_metrics(
            [
                {
                    "target": self.actual_target,
                    "prediction": json.dumps(prediction, ensure_ascii=False),
                }
            ]
        )
        self.assertEqual(metrics["matched_speaker_accuracy"], 1.0)
        self.assertEqual(metrics["sequence_exact_match"], 1.0)
        self.assertEqual(metrics["literal_sequence_exact_match"], 0.0)

    def test_invalid_json_is_excluded_from_valid_output_cer(self):
        metrics = calculate_metrics(
            [{"target": self.actual_target, "prediction": "[[{broken json"}]
        )
        self.assertEqual(metrics["json_parse_rate"], 0.0)
        self.assertEqual(metrics["cer_evaluated_sequences"], 0)
        self.assertEqual(metrics["cer_including_failures_as_deletions"], 1.0)
        for name in ("cer", "cer_valid_outputs", "dialogue_f1", "speaker_accuracy", "reading_order_accuracy"):
            self.assertIsNone(metrics[name])
        self.assertEqual(metrics["support"]["unscored_target_dialogues"], sum(map(len, self.actual_target)))

    def test_fenced_flat_list_reports_both_failures_without_scoring_it(self):
        record = {"sequence_id": "fenced", "target": [[{"speaker": "character1", "text": "HELLO"}]],
                  "prediction": '```json\n[{"speaker": "UNKNOWN", "text": "HELLO"}]\n```'}
        report = prediction_report(record)
        issues = " ".join(report["format_diagnostics"]["issues"])
        self.assertIn("Markdown code fence", issues)
        self.assertIn("Flat dialogue list", issues)
        self.assertIsNone(report["metrics"]["cer"])
        self.assertEqual(report["page_diagnostics"], [])
        self.assertIn("N/A", metric_summary_rows(report["metrics"])[1]["Value"])
        self.assertIsNone(json.loads(json.dumps(report))["metrics"]["cer"])

    def test_mixed_validity_keeps_content_scores_and_failure_penalties_separate(self):
        target = [[{"speaker": "character1", "text": "HELLO"}]]
        metrics = calculate_metrics([
            {"target": target, "prediction": json.dumps(target)},
            {"target": target, "prediction": "not json"},
        ])
        self.assertEqual(metrics["cer"], 0)
        self.assertEqual(metrics["dialogue_f1"], 1)
        self.assertEqual(metrics["schema_compliance_rate"], 0.5)
        self.assertEqual(metrics["cer_including_failures_as_deletions"], 0.5)
        self.assertAlmostEqual(metrics["dialogue_f1_including_format_failures"], 2 / 3)
        self.assertEqual(metrics["support"]["missing_dialogues"], 0)
        self.assertEqual(metrics["support"]["unscored_target_dialogues"], 1)

    def test_missing_text_is_a_real_failure_but_unmatched_speakers_are_unavailable(self):
        target = [[{"speaker": "character1", "text": "HELLO"}]]
        metrics = calculate_metrics([{"target": target, "prediction": "[[]]"}])
        self.assertEqual(metrics["cer"], 1)
        self.assertEqual(metrics["dialogue_f1"], 0)
        self.assertEqual(metrics["dialogue_recall"], 0)
        self.assertIsNone(metrics["dialogue_precision"])
        self.assertIsNone(metrics["speaker_accuracy"])
        self.assertIsNone(metrics["reading_order_accuracy"])
        self.assertEqual(metrics["support"]["missing_dialogues"], 1)

    def test_page_count_missing_lines_extras_order_and_unknown_speakers_are_visible(self):
        target = [[{"speaker": "character1", "text": "HELLO"},
                   {"speaker": "character2", "text": "GOODBYE"},
                   {"speaker": "character1", "text": "ZZZZZ"}], []]
        prediction = [[{"speaker": "UNKNOWN", "text": "GOODBYE"},
                       {"speaker": "UNKNOWN", "text": "HELLO"},
                       {"speaker": "UNKNOWN", "text": "XXXXX"}]]
        report = prediction_report({"sequence_id": "errors", "target": target,
                                    "prediction": json.dumps(prediction)})
        metrics = report["metrics"]
        self.assertEqual(metrics["speaker_accuracy"], 0)
        self.assertEqual(metrics["reading_order_accuracy"], 0)
        self.assertEqual(metrics["support"]["unknown_speakers"], 3)
        self.assertEqual(metrics["support"]["missing_pages"], 1)
        self.assertEqual(metrics["support"]["missing_dialogues"], 1)
        self.assertEqual(metrics["support"]["extra_dialogues"], 1)
        self.assertIn("Page count mismatch", report["format_diagnostics"]["issues"][0])
        page = report["page_diagnostics"][0]
        self.assertEqual(page["unmatched_target"][0]["text"], "ZZZZZ")
        self.assertEqual(page["unmatched_prediction"][0]["text"], "XXXXX")
        self.assertEqual(page["order_errors"], 1)

    def test_empty_evaluation_and_empty_transcripts_do_not_report_perfect_cer(self):
        empty = calculate_metrics([])
        self.assertIsNone(empty["cer"])
        self.assertIsNone(empty["schema_compliance_rate"])
        metrics = calculate_metrics([{"target": [[]], "prediction": "[[]]"}])
        self.assertIsNone(metrics["cer"])
        self.assertIsNone(metrics["dialogue_f1"])
        self.assertIsNone(metrics["narration_f1"])
        self.assertEqual(metrics["sequence_exact_match"], 1)

    def test_schema_error_explains_field_types_and_flat_lists(self):
        for raw, reason in (("null", "Expected a pages list"),
                            ('[{"speaker": "UNKNOWN", "text": "HELLO"}]', "Flat dialogue list"),
                            ('[[{"speaker": 1, "text": "HELLO"}]]', "string-valued")):
            with self.subTest(raw=raw):
                report = prediction_report({"sequence_id": "bad", "target": [[]], "prediction": raw})
                self.assertTrue(report["prediction_is_valid_json"])
                self.assertFalse(report["format_diagnostics"]["schema_valid"])
                self.assertIn(reason, " ".join(report["format_diagnostics"]["issues"]))

    def test_dialogue_matching_maximizes_match_count_before_similarity(self):
        # Greedy matching takes aa->aabb and leaves aaa unmatched. The optimal
        # assignment retains both valid edges at the 0.5 threshold.
        true_page = [
            {"speaker": "character1", "text": "aa"},
            {"speaker": "character1", "text": "aaa"},
        ]
        predicted_page = [
            {"speaker": "character1", "text": "aabb"},
            {"speaker": "character1", "text": "ab"},
        ]
        matches = _match_page(true_page, predicted_page, minimum_similarity=0.5)
        self.assertEqual(
            [(true_index, pred_index) for true_index, pred_index, _ in matches],
            [(0, 1), (1, 0)],
        )

    def test_cer_operations_are_decomposed(self):
        self.assertEqual(_edit_operations("ABC", "AXC"), (1, 0, 0))
        self.assertEqual(_edit_operations("ABC", "AC"), (0, 1, 0))
        self.assertEqual(_edit_operations("AC", "ABC"), (0, 0, 1))

        target = [[{"speaker": "character1", "text": "ABC"}]]
        prediction = [[{"speaker": "character1", "text": "AXC"}]]
        metrics = calculate_metrics(
            [{"target": target, "prediction": json.dumps(prediction)}]
        )
        self.assertAlmostEqual(metrics["cer_substitution_rate"], 1 / 3)
        self.assertEqual(metrics["cer_deletion_rate"], 0.0)
        self.assertEqual(metrics["cer_insertion_rate"], 0.0)

    def test_saved_prediction_can_be_inspected_and_exported_per_sequence(self):
        record = {
            "sequence_id": "example_sequence",
            "target": self.actual_target,
            "prediction": json.dumps(self.actual_target, ensure_ascii=False),
            "generated_tokens": 123,
            "reached_eos": True,
            "hit_max_new_tokens": False,
            "suspected_repetition_loop": False,
        }
        predictions_file = ROOT / "tests" / "_prediction_report_test.jsonl"
        output_file = predictions_file.with_name("per_sequence_results.jsonl")
        try:
            predictions_file.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report = inspect_prediction(
                predictions_file,
                sequence_id="example_sequence",
                print_output=False,
            )
            self.assertTrue(report["prediction_is_valid_json"])
            self.assertEqual(report["metrics"]["sequence_exact_match"], 1.0)
            self.assertEqual(
                report["generation_diagnostics"]["generated_tokens"], 123
            )

            saved_path = save_per_sequence_reports(
                predictions_file, output_file=output_file
            )
            saved_reports = [
                json.loads(line)
                for line in saved_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved_reports), 1)
            self.assertEqual(saved_reports[0]["sequence_id"], "example_sequence")
            self.assertEqual(
                saved_reports[0]["metrics"]["sequence_exact_match"], 1.0
            )
        finally:
            predictions_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
