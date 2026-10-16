import json
import sys
import shutil
import uuid
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocr_reports import get_ocr_result, write_report_index


class OCRReportTests(unittest.TestCase):
    def test_lookup_by_sample_or_sequence_and_missing_page(self):
        output = ROOT / "outputs" / ("test_reports_" + uuid.uuid4().hex)
        output.mkdir()
        try:
            entries = [dict(sequence_id="s<&", sample_index=4, page_index=0,
                            preview="sample_4_page_0.html", result="sample_4_page_0.json",
                            transcript="sample_4_page_0.txt", raw_count=2, group_count=1)]
            write_report_index(entries, output)
            (output / entries[0]["result"]).write_text(json.dumps(dict(
                transcript="HELLO WORLD", raw_ocr=[{"id": "c0"}, {"id": "c1"}]),
                ensure_ascii=False), encoding="utf-8")
            sample = get_ocr_result(ROOT, sample=4, page=0, output=output)
            sequence = get_ocr_result(ROOT, sequence_id="s<&", page=0, output=output)
            self.assertEqual(sample, sequence)
            self.assertEqual(sample["transcript"], "HELLO WORLD")
            self.assertEqual(Path(sample["preview_path"]).name, entries[0]["preview"])
            self.assertIn("s&lt;&amp;", (output / "index.html").read_text(encoding="utf-8"))
            with self.assertRaises(KeyError):
                get_ocr_result(ROOT, sample=4, page=1, output=output)
        finally:
            shutil.rmtree(output)
