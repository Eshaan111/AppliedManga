import contextlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_utils import save_jsonl, load_jsonl
from dataset_utils import save_json
from ocr_utils import sha256
from text_extraction_diagnostics import run_diagnostics


@contextlib.contextmanager
def fixture():
    root = ROOT / 'outputs' / ('test_diagnostics_' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        images = root / 'TaskData/development/images/s'
        images.mkdir(parents=True)
        for p in range(1, 4):
            (images / f'{p:02}.png').write_bytes(b'image fixture')
        paths = [f'development/images/s/{p:02}.png' for p in range(1, 4)]
        save_json([dict(sequence_id='s', split='development', images=paths)], root / 'TaskData/sequences.json')
        save_json(dict(train=[], val=['s']), root / 'outputs/split.json')
        refs = root / 'datasets/val_references.jsonl'
        save_jsonl([dict(sequence_id='s', pages=[[dict(text='HELLO', speaker='x'),
                                                dict(text='WORLD', speaker='x')], [], []])], refs)
        raw = root / 'raw.jsonl'
        fragments = [dict(id='c0', text='HELLO', score=.9, box=[0, 0, 40, 10]),
                     dict(id='c1', text='WORLD', score=.9, box=[0, 12, 40, 22]),
                     dict(id='c2', text='BAM', score=.9, box=[0, 24, 40, 34])]
        save_jsonl([dict(sequence_id='s', pages=[fragments, [], []])], raw)
        output = root / 'pipeline'
        # Intentionally separate nearby lines: diagnostics must not rejoin them.
        groups = [dict(id=f'g{i}', text=c['text'], box=c['box'], candidate_ids=[c['id']], score=c['score'])
                  for i, c in enumerate(fragments[:2])]
        candidates = output / 'candidates.jsonl'
        save_jsonl([dict(sequence_id='s', pages=[groups, [], []])], candidates)
        audit = output / 'filter_audit.jsonl'
        save_jsonl([dict(sequence_id='s', page_index=p,
                        excluded=[dict(candidate=fragments[2], reason='sound_effect_word')] if p == 0 else [])
                    for p in range(3)], audit)
        save_json(dict(split='val', sequence_ids=['s'], output_sha256=sha256(candidates),
                       audit_sha256=sha256(audit), split_sha256=sha256(root / 'outputs/split.json'),
                       images={'s': [sha256(root / 'TaskData' / p) for p in paths]},
                       raw_sources=[dict(path=str(raw), sha256=sha256(raw))]),
                  candidates.with_suffix('.metadata.json'))
        yield root, candidates
    finally:
        shutil.rmtree(root)


class DiagnosticTests(unittest.TestCase):
    def test_saved_groups_preserved_and_references_only_affect_diagnostics(self):
        with fixture() as (root, candidates):
            before = candidates.read_bytes()
            with patch('ocr_joining.join_ocr_candidates', side_effect=AssertionError('Must not join')):
                summary = run_diagnostics(root, candidates, root / 'diagnostics', render_previews=False)
            self.assertEqual(summary['proposed_groups'], 2)
            self.assertEqual(summary['excluded_sound_effect_fragments'], 1)
            self.assertEqual(summary['matched_targets'], 2)
            report = load_jsonl(root / 'diagnostics/allocations.jsonl')[0]
            self.assertEqual([g['text'] for g in report['groups']], ['HELLO', 'WORLD'])
            refs = root / 'datasets/val_references.jsonl'
            save_jsonl([dict(sequence_id='s', pages=[[], [], []])], refs)
            changed = run_diagnostics(root, candidates, root / 'diagnostics', render_previews=False)
            self.assertEqual(changed['matched_targets'], 0)
            self.assertEqual(changed['proposed_groups'], 2)
            self.assertEqual(candidates.read_bytes(), before)

    def test_changed_candidates_rejected_before_output(self):
        with fixture() as (root, candidates):
            candidates.write_bytes(candidates.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_diagnostics(root, candidates, root / 'diagnostics', render_previews=False)
            self.assertFalse((root / 'diagnostics').exists())

    def test_missing_upstream_and_overlapping_output_rejected(self):
        with fixture() as (root, candidates):
            with self.assertRaisesRegex(ValueError, 'separate'):
                run_diagnostics(root, candidates, candidates.parent, render_previews=False)
            with self.assertRaisesRegex(FileNotFoundError, 'Run TextExtractionPipeline'):
                run_diagnostics(root, root / 'missing/candidates.jsonl', root / 'diagnostics', render_previews=False)
