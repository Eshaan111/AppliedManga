"""Standalone inputs, OCR-cache isolation, and label-independent joining."""
import json
import sys
import contextlib
import shutil
import uuid
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_jsonl, save_jsonl
from directory_ocr import inspect_directory, run_directory


@contextlib.contextmanager
def fixture_directory():
    # Plain mkdir avoids restrictive temporary-directory ACLs on this Windows host.
    path = (ROOT / "outputs" / ("test_directory_ocr_" + uuid.uuid4().hex)).resolve()
    assert path.is_relative_to((ROOT / "outputs").resolve())
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class DirectoryOCRTests(unittest.TestCase):
    def make_input(self, root, name="new_data"):
        from PIL import Image
        source = root / name
        folder = source / "images/seq_example"
        folder.mkdir(parents=True)
        for page in range(1, 4):
            Image.new("RGB", (100, 100), "white").save(folder / f"{page:02d}.png")
        save_jsonl([dict(sequence_id="seq_example", pages=[
            [dict(speaker="character1", text="HELLO WORLD")], [], []])], source / "labels.jsonl")
        return source

    def test_join_reports_resume_and_labels_do_not_change_joined_text(self):
        with fixture_directory() as root:
            source = self.make_input(root)
            output = root / "JOINED_OCR"
            calls = []
            class FakeOCR:
                def predict(self, path):
                    calls.append(path)
                    return [types.SimpleNamespace(json={"res": dict(
                        rec_texts=["HELLO", "WORLD"], rec_scores=[.99, .98],
                        rec_boxes=[[10, 10, 60, 20], [10, 24, 60, 34]])})]
            summary = run_directory(source, output, ocr_factory=FakeOCR)
            self.assertEqual(len(calls), 3)
            self.assertEqual(summary["proposed_groups"], 3)
            self.assertEqual(summary["matched_targets"], 1)
            joined = load_jsonl(output / "joined.jsonl")
            for page in joined[0]["pages"]:
                self.assertEqual(page[0]["text"], "HELLO WORLD")
                self.assertEqual(page[0]["candidate_ids"], ["c0", "c1"])
                self.assertNotIn("target_text", page[0])
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "sample_0_page_0_groups.png").is_file())
            self.assertEqual(json.loads((output / "sample_0_page_0.json").read_text())
                             ["transcript"], "HELLO WORLD")
            # A reference revision rebuilds comparisons without changing OCR or groups.
            save_jsonl([dict(sequence_id="seq_example", pages=[
                [dict(speaker="character1", text="TOTALLY DIFFERENT")], [], []])], source / "labels.jsonl")
            summary = run_directory(source, output, ocr_factory=lambda: self.fail("Cache should be reused"))
            self.assertEqual(load_jsonl(output / "joined.jsonl"), joined)
            self.assertEqual(summary["matched_targets"], 0)

            # Even identical sequence IDs from another source cannot reuse this cache.
            other = self.make_input(root, "other_data")
            with self.assertRaisesRegex(ValueError, "different or changed images"):
                run_directory(other, output, ocr_factory=FakeOCR)
            from PIL import Image
            Image.new("RGB", (100, 100), "black").save(source / "images/seq_example/01.png")
            with self.assertRaisesRegex(ValueError, "different or changed images"):
                run_directory(source, output, ocr_factory=FakeOCR)

    def test_invalid_coverage_and_duplicate_ids_fail_before_output(self):
        with fixture_directory() as root:
            source = self.make_input(root)
            labels = load_jsonl(source / "labels.jsonl")
            save_jsonl(labels * 2, source / "labels.jsonl")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                run_directory(source, root / "out")
            self.assertFalse((root / "out").exists())
            save_jsonl(labels, source / "labels.jsonl")
            (source / "images/unlabelled").mkdir()
            with self.assertRaisesRegex(ValueError, "Images/labels disagree"):
                inspect_directory(source)

    def test_existing_candidates_skip_uncached_inputs_without_ocr_or_source_writes(self):
        with fixture_directory() as root:
            source = self.make_input(root)
            shutil.copytree(source / "images/seq_example", source / "images/seq_uncached")
            labels = load_jsonl(source / "labels.jsonl")
            save_jsonl(labels + [dict(sequence_id="seq_uncached", pages=[[], [], []])],
                       source / "labels.jsonl")
            candidates = root / "candidates.jsonl"
            save_jsonl([dict(sequence_id="seq_example", pages=[[
                dict(id="c0", text="HELLO", box=[10, 10, 60, 20]),
                dict(id="c1", text="WORLD", box=[10, 24, 60, 34])], [], []])], candidates)
            before = candidates.read_bytes()
            output = root / "joined"
            summary = run_directory(source, output, candidates_path=candidates,
                                    ocr_factory=lambda: self.fail("Must never run OCR"))
            self.assertEqual((summary["sequences"], summary["pages"], summary["uncached_sequences"]),
                             (1, 3, 1))
            self.assertEqual(candidates.read_bytes(), before)
            self.assertFalse((output / "candidates.jsonl").exists())
            joined = load_jsonl(output / "joined.jsonl")
            self.assertEqual(joined[0]["pages"][0][0]["text"], "HELLO WORLD")
            self.assertEqual(joined[0]["pages"][1:], [[], []])
            run_directory(source, output, candidates_path=candidates,
                          ocr_factory=lambda: self.fail("Must never run OCR on rerun"))
            # An absent OCR page is malformed, rather than an empty page.
            save_jsonl([dict(sequence_id="seq_example", pages=[[], []])], candidates)
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                run_directory(source, root / "bad", candidates_path=candidates)
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
