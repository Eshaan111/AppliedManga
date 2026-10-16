"""Regression tests for unseen inputs, cache invalidation, and preserved OCR."""
import ast
import contextlib
import json
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ocr_utils import discover_sequences, extract_sequences
from ocr_joining import join_ocr_candidates


@contextlib.contextmanager
def fixture():
    path = ROOT / 'outputs' / ('test_ocr_' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class OCRTests(unittest.TestCase):
    def source(self, path):
        folder = path / 'input/images/new_character'
        folder.mkdir(parents=True)
        for index in range(1, 4):
            (folder / f'{index:02}.png').write_bytes(f'image{index}'.encode())
        return path / 'input'

    def engine(self):
        calls = []
        def predict(path):
            calls.append(path)
            return [types.SimpleNamespace(json={'res': dict(
                rec_texts=['HELLO', 'WORLD'], rec_scores=[.99, .98],
                rec_boxes=[[0, 0, 50, 10], [0, 12, 50, 22]])})]
        return types.SimpleNamespace(predict=predict), calls

    def test_unseen_data_and_labels_do_not_change_extraction_or_joining(self):
        with fixture() as path:
            source = self.source(path)
            rows = discover_sequences(source)
            engine, calls = self.engine()
            first = extract_sequences(rows, engine, path / 'first.jsonl', {'engine': 'fixture'})
            (source / 'labels.jsonl').write_text('INVALID LABELS ARE NEVER READ', encoding='utf8')
            second = extract_sequences(discover_sequences(source), engine, path / 'second.jsonl', {'engine': 'fixture'})
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 6)
            groups = join_ocr_candidates(first['new_character']['pages'][0])
            self.assertEqual(groups[0]['text'], 'HELLO WORLD')
            self.assertEqual(groups[0]['candidate_ids'], ['c0', 'c1'])

    def test_resume_and_changed_images_runtime_or_output_are_rejected(self):
        with fixture() as path:
            rows = discover_sequences(self.source(path))
            engine, calls = self.engine()
            output = path / 'candidates.jsonl'
            runtime = {'engine': 'fixture'}
            extract_sequences(rows, engine, output, runtime)
            original = output.read_bytes()
            extract_sequences(rows, engine, output, runtime)
            self.assertEqual(len(calls), 3)
            self.assertEqual(output.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'Runtime'):
                extract_sequences(rows, engine, output, {'engine': 'changed'})
            rows[0]['images'][0].write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError, 'images changed'):
                extract_sequences(rows, engine, output, runtime)
            output.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                extract_sequences(rows, engine, output, runtime)

    def test_legacy_cache_is_preserved(self):
        with fixture() as path:
            rows = discover_sequences(self.source(path))
            output = path / 'candidates.jsonl'
            output.write_text('legacy data', encoding='utf8')
            engine, calls = self.engine()
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                extract_sequences(rows, engine, output, {})
            self.assertEqual(output.read_text(encoding='utf8'), 'legacy data')
            self.assertEqual(calls, [])

    def test_manifest_order_duplicate_ids_and_unsafe_paths(self):
        with fixture() as path:
            source = self.source(path)
            entry = dict(sequence_id='new_character', split='test',
                         images=[f'images/new_character/{i:02}.png' for i in (3, 1, 2)])
            manifest = source / 'sequences.json'
            manifest.write_text(json.dumps([entry]), encoding='utf8')
            self.assertEqual([p.name for p in discover_sequences(source, 'test')[0]['images']],
                             ['03.png', '01.png', '02.png'])
            manifest.write_text(json.dumps([entry, entry]), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                discover_sequences(source)
            entry['images'][0] = '../outside.png'
            manifest.write_text(json.dumps([entry]), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                discover_sequences(source)

    def test_retained_notebook_code_compiles(self):
        for path in (ROOT / 'notebooks').rglob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8-sig'))
            self.assertEqual(notebook['nbformat'], 4)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']), filename=path.name)

if __name__ == '__main__':
    unittest.main()
