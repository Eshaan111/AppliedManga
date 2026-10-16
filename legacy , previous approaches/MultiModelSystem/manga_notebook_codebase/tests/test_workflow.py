import contextlib
import json
import sys
import shutil
import uuid
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_jsonl, save_jsonl
from dataset_utils import prepare_datasets, read_dataset, save_json
from eval_utils import (speaker_accuracy, pairwise_order_accuracy, identity_switch_metrics,
                        empty_page_metrics, matched_lines, narration_metrics, validate_prediction_schema)


@contextlib.contextmanager
def fixture_directory():
    path = (ROOT / "outputs" / ("test_fixture_" + uuid.uuid4().hex)).resolve()
    assert path.is_relative_to((ROOT / "outputs").resolve())
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class WorkflowTests(unittest.TestCase):
    def test_persisted_split_and_portable_triplets(self):
        split = prepare_datasets(ROOT)
        self.assertEqual((len(split["train"]), len(split["val"])), (64, 16))
        self.assertFalse(set(split["train"]) & set(split["val"]))
        before = (ROOT / "outputs/split.json").read_bytes()
        prepare_datasets(ROOT)
        self.assertEqual(before, (ROOT / "outputs/split.json").read_bytes())
        for name in ("train", "val", "test"):
            rows = read_dataset(ROOT, name)
            for row in rows:
                self.assertEqual(len(row["images"]), 3)
                self.assertTrue(all((ROOT / "TaskData" / p).is_file() for p in row["images"]))
                if name == "test":
                    self.assertNotIn("target", row)

    def test_order_task_has_shuffled_content_ids(self):
        tasks = load_jsonl(ROOT / "datasets/components/reading_order.jsonl")
        self.assertEqual(len(tasks), 48)
        self.assertTrue(any(t["expected"] != [c["id"] for c in t["candidates"]] for t in tasks))
        for task in tasks:
            self.assertEqual(set(task["expected"]), {x["id"] for x in task["candidates"]})

    def test_speaker_optimum_handles_extra_and_many_identities(self):
        ref = ["character1"] * 4 + ["character2"] * 4 + ["character1"]
        pred = ["z"] * 4 + ["y"] * 4 + ["a"]
        self.assertAlmostEqual(speaker_accuracy(ref, pred), 8 / 9)
        many = [f"character{i}" for i in range(12)] * 2
        renamed = [f"other{i}" for i in range(12)] * 2
        self.assertEqual(speaker_accuracy(many, renamed), 1)
        self.assertEqual(speaker_accuracy(["UNKNOWN", "character1"], ["anything", "x"]), 1)
        self.assertEqual(narration_metrics(["UNKNOWN", "NARRATION"], ["NARRATION", "NARRATION"])["count"], 1)

    def test_incomplete_orders_are_errors(self):
        self.assertEqual(pairwise_order_accuracy(["a", "b", "c"], ["c", "b", "a"]), 0)
        for order in (["a"], ["a", "a", "c"], ["a", "b", "wrong"]):
            with self.assertRaises(ValueError):
                pairwise_order_accuracy(["a", "b", "c"], order)

    def test_identity_and_missing_empty_page(self):
        matches = [(0, "character1", "x"), (1, "character1", "x"), (2, "character1", "y")]
        self.assertEqual(identity_switch_metrics(matches)["identity_switch_rate"], 0.5)
        row = dict(sequence_id="s", pages=[[], [], []])
        self.assertEqual(empty_page_metrics([row], [])["empty_page_accuracy"], 0)
        self.assertEqual(empty_page_metrics([row], [row])["empty_page_accuracy"], 1)


    def test_duplicate_text_line_matching(self):
        a = [dict(speaker="x", text="Hi"), dict(speaker="y", text="Hi")]
        pairs = matched_lines(a, a)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({i for i, _ in pairs}), 2)

    def test_ocr_nested_result_and_polygon_boxes(self):
        from ocr_utils import ocr_page
        class FakeOCR:
            def predict(self, path):
                return [types.SimpleNamespace(json={"res": {
                    "rec_texts": ["HELLO"], "rec_scores": [0.95],
                    "rec_polys": [[[1, 2], [9, 2], [9, 8], [1, 8]]]}})]
        self.assertEqual(ocr_page(FakeOCR(), "image")[0]["box"], [1, 2, 9, 8])


    def test_geometric_joining_separates_columns_and_keeps_all_ids(self):
        from ocr_joining import join_ocr_candidates, compare_groups_to_targets
        candidates = [dict(id="c3", text="WORLD", box=[100, 22, 150, 32]),
                      dict(id="c0", text="WELL,", box=[0, 10, 50, 20]),
                      dict(id="c2", text="SHIT.", box=[5, 22, 45, 32]),
                      dict(id="c1", text="HELLO", box=[100, 10, 150, 20]),
                      dict(id="c4", text="BAM", box=[0, 100, 70, 150])]
        groups = join_ocr_candidates(candidates)
        self.assertEqual({g['text'] for g in groups}, {"WELL, SHIT.", "HELLO WORLD", "BAM"})
        self.assertEqual(sorted(cid for g in groups for cid in g['candidate_ids']), sorted(c['id'] for c in candidates))
        self.assertEqual(join_ocr_candidates(list(reversed(candidates))), groups)
        targets = [dict(text="WELL, SHIT.", speaker="x"), dict(text="HELLO WORLD", speaker="y")]
        allocation = compare_groups_to_targets(groups, targets)
        self.assertEqual(len([g for g in allocation['groups'] if g['target_index'] is not None]), 2)
        self.assertEqual(allocation['unmatched_targets'], [])
        self.assertEqual([g['text'] for g in allocation['groups']], [g['text'] for g in groups])
        self.assertEqual(join_ocr_candidates([]), [])








if __name__ == "__main__":
    unittest.main()
