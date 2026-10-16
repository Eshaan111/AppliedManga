"""Regression checks for automatic crops, empty pages, and label-free inference."""
import json
from pathlib import Path
import sys
from contextlib import contextmanager
import uuid
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detection
import automatic_identity as auto


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parent / ('fixture_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        for path in sorted(root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        root.rmdir()


class AutomaticTests(unittest.TestCase):
    def test_expansion_clips_and_rounds_outward(self):
        self.assertEqual(detection.expand_box([0, 0, 10, 20], (30, 20), 1.8), [0, 0, 14, 28])
        with self.assertRaises(ValueError):
            detection.expand_box([0, 0, 1, 1], (20, 20), float('nan'))
        predictions = [dict(box=[-2, -2, 12, 12], score=.9),
                       dict(box=[0, 0, 2, 2], score=.1),
                       dict(box=[20, 20, 30, 30], score=.9),
                       dict(box=[float('nan'), 0, 2, 2], score=.9)]
        result = detection.normalize_predictions(predictions, (10, 10))
        self.assertEqual(result, [dict(box=[0, 0, 10, 10], score=.9, label='face')])

    def test_detection_to_identity_needs_no_labels_and_keeps_empty_pages(self):
        with fixture_directory() as temp:
            root = Path(temp)
            image = root / 'page.png'
            Image.new('RGB', (30, 40), 'white').save(image)
            pages = [dict(sequence_id='s', page_index=i, split='calibration', image_path=str(image)) for i in range(3)]
            predictions = iter([[dict(box=[2, 3, 12, 13], score=.9)], [],
                                [dict(box=[2, 3, 12, 13], score=.95)]])
            manifest = detection.run_detection(pages, root / 'detect', predictor=lambda _: next(predictions))
            self.assertEqual(len(manifest['rows']), 2)
            self.assertTrue(all(r['identity_label'] is None for r in manifest['rows']))
            self.assertTrue((root / 'detect/overlays/s_p1.png').exists())
            self.assertEqual(len(auto.load_crops(root / 'detect/detections.json', 2)[1]), 1)
            with patch.object(auto.dino, 'load_model', return_value=(None, None, 'test')), \
                 patch.object(auto.dino, 'embed', return_value=np.tile([1., 0.], (2, 1))):
                summary = auto.run_identity(root / 'detect/detections.json', root / 'identity')
            self.assertEqual(summary['cross_page_pairs'], 1)
            pairs = json.loads((root / 'identity/cross_page_pairs.json').read_text())
            self.assertEqual(pairs[0]['similarity'], 1.)
            self.assertEqual(pairs[0]['same_character'], '')
            self.assertEqual((pairs[0]['page1'], pairs[0]['page2']), (0, 2))
            self.assertTrue((root / 'identity/pair_review.csv').exists())
            self.assertTrue((root / 'identity/pair_review.html').exists())
            self.assertFalse((root / 'identity/clusters').exists())
            self.assertFalse((root / 'identity/character_results.jsonl').exists())
            self.assertFalse((root / 'identity/evaluation.json').exists())

    def test_no_detections_fails_before_model_loading(self):
        with fixture_directory() as temp:
            root = Path(temp)
            image = root / 'page.png'
            Image.new('RGB', (20, 20)).save(image)
            detection.run_detection([dict(sequence_id='s', page_index=0, image_path=str(image))],
                                    root / 'detect', predictor=lambda _: [])
            with patch.object(auto.dino, 'load_model') as loader:
                with self.assertRaisesRegex(ValueError, 'No detected crops'):
                    auto.run_identity(root / 'detect/detections.json', root / 'identity')
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
