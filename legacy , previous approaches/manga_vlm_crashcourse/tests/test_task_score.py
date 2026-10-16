import json
import uuid
from contextlib import contextmanager
import unittest
from pathlib import Path

from Scripts.taskScore import score_saved_predictions
from TaskData import score


@contextmanager
def workspace_temporary_directory():
    folder = Path(__file__).resolve().parent / f'_score_test_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        yield folder
    finally:
        for file in folder.rglob('*'):
            if file.is_file():
                file.unlink()
        for directory in sorted((p for p in folder.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            directory.rmdir()
        folder.rmdir()


class TaskScoreTests(unittest.TestCase):
    def test_official_score_roundtrip_and_invalid_output_penalty(self):
        pages = [[{'speaker': 'character1', 'text': 'HELLO!'}], [], []]
        with workspace_temporary_directory() as temporary:
            folder = Path(temporary)
            labels = folder / 'labels.jsonl'
            labels.write_text('\n'.join(json.dumps({'sequence_id': key, 'pages': pages}) for key in ['good', 'bad', 'unused']), encoding='utf-8')
            predictions = folder / 'predictions.jsonl'
            predictions.write_text('\n'.join(json.dumps({'sequence_id': key, 'prediction': raw}) for key, raw in [('good', json.dumps(pages)), ('bad', '```json\n[]\n```')]), encoding='utf-8')
            output = folder / 'output'
            report = score_saved_predictions(predictions, labels, ['good', 'bad'], output)
            expected = score.evaluate(score.load_jsonl(output / 'references.jsonl', reference=True), score.load_jsonl(output / 'submission.jsonl'))
            self.assertEqual(report, expected)
            self.assertEqual(report['missing_sequences'], ['bad'])
            self.assertEqual(report['macro']['text_order_score'], 0.5)
            self.assertEqual(report['macro']['balanced_joint_f1'], 0.5)
            self.assertEqual(set(report['sequences']), {'good', 'bad'})
            self.assertEqual(json.loads((output / 'invalid_predictions.json').read_text())[0]['sequence_id'], 'bad')

    def test_duplicate_or_unexpected_prediction_ids_are_rejected(self):
        with workspace_temporary_directory() as temporary:
            folder = Path(temporary)
            labels = folder / 'labels.jsonl'
            labels.write_text(json.dumps({'sequence_id': 'a', 'pages': [[], [], []]}), encoding='utf-8')
            predictions = folder / 'predictions.jsonl'
            for keys in [['a', 'a'], ['other']]:
                predictions.write_text('\n'.join(json.dumps({'sequence_id': key, 'prediction': '[[],[],[]]'}) for key in keys), encoding='utf-8')
                with self.assertRaises(ValueError):
                    score_saved_predictions(predictions, labels, ['a'], folder / 'output')
