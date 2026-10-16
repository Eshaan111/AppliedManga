"""Checks for sequence isolation, evaluation leakage, and centroid matching."""
import json
import sys
import unittest
import uuid
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import identity as dino


def row(sid, page, label, strategy='face', split='calibration', cid=None):
    return dict(sequence_id=sid, page_index=page, identity_label=label,
                crop_id=cid or f'p{page}_c0', strategy=strategy, split=split)


class IdentityTests(unittest.TestCase):
    def test_pairs_exclude_same_page_other_sequence_and_other_style(self):
        rows = [row('s1', 0, 'A'), row('s1', 0, 'B', cid='p0_c1'),
                row('s1', 1, 'A'), row('s2', 1, 'A'), row('s1', 1, 'A', strategy='head_hair')]
        vectors = np.tile([1., 0.], (len(rows), 1))
        pairs = dino.make_pairs(rows, vectors)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p['same_character'] for p in pairs}, {True, False})
        self.assertTrue(all(p['sequence_id'] == 's1' and p['strategy'] == 'face' for p in pairs))

    def test_unknown_labels_do_not_create_ground_truth_pairs(self):
        rows = [row('s', 0, None), row('s', 1, 'A')]
        self.assertEqual(dino.make_pairs(rows, np.eye(2)), [])

    def test_threshold_is_calibration_only_and_covers_negative_scores(self):
        pairs = [dict(split='calibration', similarity=x, same_character=y)
                 for x, y in [(-.8, False), (-.6, False), (-.2, True), (.2, True)]]
        report = dino.calibrate(pairs)
        self.assertLess(report['threshold'], 0)
        self.assertEqual(report['calibration']['balanced_accuracy'], 1.)
        with self.assertRaises(ValueError):
            dino.calibrate(pairs + [dict(split='heldout', similarity=.1, same_character=True)])

    def test_centroid_updates_and_identity_numbers_restart(self):
        angles = np.radians([0, 40, 55, 180])
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        rows = [row('s1', i, 'unused') for i in range(3)] + [row('s2', 0, 'unused')]
        assignments = dino.cluster(rows, vectors, .75)
        # Third vector misses first exemplar but matches updated mean centroid.
        self.assertEqual(list(assignments.values()), ['char1', 'char1', 'char1', 'char1'])

    def test_invalid_boxes_and_duplicate_sequences_rejected(self):
        root = Path(__file__).resolve().parent / ('fixture_' + uuid.uuid4().hex)
        root.mkdir()
        try:
            Image.new('RGB', (30, 40)).save(root / 'page.png')
            seq = dict(sequence_id='s', split='calibration', pages=[dict(page_index=0, image='page.png',
                characters=[dict(crop_id='p0_c0', identity_label='A', boxes={'face': [0, 0, 31, 20]})])])
            data = dict(schema_version=1, task_data_root=str(root), sequences=[seq])
            path = root / 'annotations.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Invalid face box'):
                dino.load_annotations(path)
            seq['pages'][0]['characters'][0]['boxes']['face'] = [0, 0, 20, 20]
            data['sequences'].append(dict(seq, split='heldout'))
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Duplicate sequence'):
                dino.load_annotations(path)
        finally:
            for filename in ('page.png', 'annotations.json'):
                (root / filename).unlink(missing_ok=True)
            root.rmdir()

    def test_metrics_distinguish_empty_and_single_class_evaluations(self):
        self.assertEqual(dino.metrics([], .5), {'count': 0})
        report = dino.metrics([dict(similarity=.8, same_character=True)], .5)
        self.assertIsNone(report['balanced_accuracy'])
        self.assertIsNone(report['roc_auc'])
        self.assertEqual(report['accuracy'], 1.)

    def test_complete_export_preserves_oracle_label_separation(self):
        root = Path(__file__).resolve().parent / ('fixture_' + uuid.uuid4().hex)
        root.mkdir()
        try:
            Image.new('RGB', (30, 40)).save(root / 'page.png')
            sequences = []
            for sid, split in (('s1', 'calibration'), ('s2', 'heldout')):
                pages = [dict(page_index=p, image='page.png', characters=[
                    dict(crop_id=f'p{p}_c{c}', identity_label=label, boxes={'face': [0, 0, 20, 20]})
                    for c, label in enumerate(('A', 'B'))]) for p in range(2)]
                sequences.append(dict(sequence_id=sid, split=split, pages=pages))
            annotation = root / 'characters.json'
            dino.save_json(annotation, dict(schema_version=1, task_data_root=str(root), sequences=sequences))
            original_export = dino.export_crops
            with (patch.object(dino, 'load_model', return_value=(None, None, 'test')),
                    patch.object(dino, 'embed', return_value=np.tile(np.eye(2), (4, 1))),
                    patch.object(dino, 'export_crops', side_effect=lambda rows: original_export(rows, root / 'crops'))):
                report = dino.run_experiment(annotation, root / 'run')
            self.assertEqual(report['face']['heldout']['accuracy'], 1.)
            result = (root / 'run/character_results.jsonl').read_text()
            self.assertNotIn('identity_label', result)
            self.assertEqual(len(result.splitlines()), 4)
            for line in result.splitlines():
                self.assertEqual({c['identity'] for c in json.loads(line)['characters']}, {'char1', 'char2'})
            with self.assertRaises(FileExistsError):
                dino.run_experiment(annotation, root / 'run')
        finally:
            # Delete only files and empty directories under this explicit test fixture.
            for path in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            root.rmdir()


if __name__ == '__main__':
    unittest.main()
