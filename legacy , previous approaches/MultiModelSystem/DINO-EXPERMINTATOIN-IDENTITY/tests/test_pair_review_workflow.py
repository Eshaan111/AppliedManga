"""Checks for human label preservation and score-independent review sampling."""
import json
from pathlib import Path
import sys
from contextlib import contextmanager
import uuid
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
import pair_review_workflow as review


@contextmanager
def fixture_directory():
    root = Path(__file__).resolve().parent / ('review_fixture_' + uuid.uuid4().hex)
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


def pair(sid, index, score=.5, label=''):
    row = dict(run_id='test-run', sequence_id=sid, strategy='face', crop1=f'p0_c{index}',
               crop2=f'p1_c{index}', similarity=str(score), same_character=label, notes='',
               split='calibration')
    row.update({tag: '' for tag in review.TAGS})
    return row


class PairReviewTests(unittest.TestCase):
    def test_sampling_ignores_scores_and_covers_sequences(self):
        rows = [pair(sid, i) for sid in ('a', 'b', 'c') for i in range(40)]
        selected = review.sample_pairs(rows, 75, 42)
        changed = [dict(r, similarity=str((i % 10) / 10)) for i, r in enumerate(rows)]
        second = review.sample_pairs(list(reversed(changed)), 75, 42)
        self.assertEqual([review.pair_key(r) for r in selected], [review.pair_key(r) for r in second])
        self.assertEqual(review.coverage(selected)['candidates'], 75)
        self.assertEqual({sid: sum(r['sequence_id'] == sid for r in selected) for sid in ('a', 'b', 'c')},
                         dict(a=25, b=25, c=25))
        self.assertEqual(len(review.sample_pairs(rows, 200)), 120)

    def test_import_preserves_scores_and_multiline_notes(self):
        with fixture_directory() as temp:
            root = Path(temp)
            rows = [pair('s', 0), pair('s', 1)]
            review.write_csv(rows, root / 'candidates.csv')
            edited = [dict(rows[1], notes='Uncertain, "hidden"\nbehind panel'),
                      dict(rows[0], same_character='same', similarity='.99', profile='yes')]
            review.write_csv(edited, root / 'download.csv')
            review.import_labels(root / 'candidates.csv', root / 'download.csv', root / 'labeled.csv')
            imported = review.read_csv(root / 'labeled.csv')
            self.assertEqual(imported[0]['similarity'], '0.5')
            self.assertEqual(imported[0]['same_character'], 'same')
            self.assertEqual(imported[1]['notes'], edited[0]['notes'])
            self.assertEqual(imported[1]['same_character'], '')
            self.assertEqual(review.read_csv(root / 'candidates.csv')[0]['same_character'], '')
            with self.assertRaises(FileExistsError):
                review.write_csv(imported, root / 'labeled.csv')
            other_run = [dict(r, run_id='different-run') for r in edited]
            review.write_csv(other_run, root / 'wrong.csv')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                review.import_labels(root / 'candidates.csv', root / 'wrong.csv', root / 'bad.csv')

    def test_invalid_and_duplicate_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            review.validate_rows([pair('s', 0, label='probably same')])
        with self.assertRaises(ValueError):
            review.validate_rows([pair('s', 0), pair('s', 0)])
        # Crop IDs reused in another run remain distinct.
        review.validate_rows([pair('s', 0), dict(pair('s', 0), run_id='second-run')])

    def test_existing_smoke_run_has_page_context_and_blind_review(self):
        run = Path(__file__).resolve().parents[1] / 'outputs/two_model_pairs'
        if not run.is_dir():
            self.skipTest('Existing smoke artifacts not available')
        before = (run / 'pair_review.csv').read_bytes()
        rows = review.load_run(run)
        self.assertEqual(len(rows), 7)
        for row in rows:
            for side in (1, 2):
                for field in ('crop_path', 'page_path', 'overlay_path'):
                    self.assertTrue(Path(row[f'{field}{side}']).is_file())
        with fixture_directory() as temp:
            path = review.write_blind_html(rows, temp / 'review.html')
            document = path.read_text(encoding='utf-8')
            self.assertEqual(document.count('<article data-index='), 7)
            self.assertIn('Download labeled CSV', document)
            self.assertIn('row.same_character?', document)
            self.assertNotIn('cosine 0.', document.split('<script>')[0])
        self.assertEqual((run / 'pair_review.csv').read_bytes(), before)

    def test_analysis_excludes_blanks_and_keeps_threshold(self):
        rows = [pair('s', 0, .9, 'same'), pair('s', 1, .7, 'same'),
                pair('s', 2, .8, 'different'), pair('s', 3, .2, 'different'),
                pair('s', 4, 1., '')]
        with fixture_directory() as temp:
            root = Path(temp) / 'analysis'
            result = review.analyze_scores(rows, root)
            self.assertEqual(result['confusion'], dict(tp=1, fn=1, fp=1, tn=1))
            self.assertEqual(result['coverage']['labeled'], 4)
            self.assertEqual(result['fixed_threshold'], .8)
            self.assertEqual(result['empirical_auc'], .75)
            self.assertEqual(len(review.read_csv(root / 'threshold_disagreements.csv')), 2)
            self.assertTrue((root / 'same_vs_different.png').is_file())
            with self.assertRaises(ValueError):
                review.analyze_scores([pair('s', 0, .9, 'same')], Path(temp) / 'empty-class')

    def test_notebook_has_executable_cells_without_results(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/04_pair_review_and_score_analysis.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(path), 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])


if __name__ == '__main__':
    unittest.main()
