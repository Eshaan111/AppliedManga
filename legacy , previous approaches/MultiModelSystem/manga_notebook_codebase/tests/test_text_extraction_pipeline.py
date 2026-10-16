import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from text_extraction_pipeline import filter_candidates, select_rows
from ocr_joining import join_ocr_candidates


class PipelineTests(unittest.TestCase):
    def candidate(self, index, text, box=None, score=.9):
        return dict(id=f'c{index}', text=text, score=score,
                    box=box if box is not None else [0, index * 12, 40, index * 12 + 10])

    def test_whole_sound_words_removed_before_joining(self):
        raw = [self.candidate(0, 'HELLO'), self.candidate(1, 'bam!!'), self.candidate(2, 'WORLD')]
        kept, excluded = filter_candidates(raw)
        self.assertEqual([c['text'] for c in kept], ['HELLO', 'WORLD'])
        self.assertEqual(excluded[0]['reason'], 'sound_effect_word')
        groups = join_ocr_candidates(kept)
        self.assertNotIn('c1', [cid for g in groups for cid in g['candidate_ids']])
        self.assertEqual(raw[1]['text'], 'bam!!')

    def test_dialogue_punctuation_numbers_and_vocalisations_preserved(self):
        texts = ['BAMBOO', 'A loud BANG!', '...', '?!', '!', 'AH!', 'UGH', '12', 'HELLO', 'HELLO']
        kept, excluded = filter_candidates([self.candidate(i, text) for i, text in enumerate(texts)])
        self.assertEqual([c['text'] for c in kept], texts)
        self.assertEqual(excluded, [])

    def test_duplicate_cleanup_keeps_stronger_detection(self):
        raw = [self.candidate(0, 'HELLO', [0, 0, 40, 10], .5),
               self.candidate(1, 'HELLO', [0, 0, 40, 10], .9), self.candidate(2, ' ')]
        kept, excluded = filter_candidates(raw)
        self.assertEqual([c['id'] for c in kept], ['c1'])
        self.assertEqual({e['reason'] for e in excluded}, {'blank_text', 'exact_duplicate'})

    def test_non_english_fragments_removed_before_joining(self):
        raw = [self.candidate(0, 'HELLO'), self.candidate(1, '魚定食'),
               self.candidate(2, 'HI あん'), self.candidate(3, 'WORLD')]
        kept, excluded = filter_candidates(raw)
        self.assertEqual([c['text'] for c in kept], ['HELLO', 'WORLD'])
        self.assertTrue(all(e['reason'] == 'non_english_fragment' for e in excluded))
        groups = join_ocr_candidates(kept)
        self.assertFalse({'c1', 'c2'} & {cid for g in groups for cid in g['candidate_ids']})
        # English quotes/ellipses and original fragment text are preserved.
        text = '“I’m here…”'
        self.assertEqual(filter_candidates([self.candidate(0, text)])[0][0]['text'], text)

    def test_new_sound_variants_and_repeats_removed(self):
        texts = ['DOINK!', 'WHOSsH', 'K-CHAK', 'PINGGGG', 'TAP TAP',
                 'SFX: CLICK', 'SFX: SCREECH', 'THE TAP IS BROKEN.']
        kept, excluded = filter_candidates([self.candidate(i, text) for i, text in enumerate(texts)])
        self.assertEqual([c['text'] for c in kept], ['THE TAP IS BROKEN.'])
        self.assertEqual(len(excluded), 7)

    def test_page_numbers_use_position_and_box_size(self):
        raw = [self.candidate(0, '12', [20, 940, 50, 960]),
               self.candidate(1, '12', [200, 400, 230, 420]),
               self.candidate(2, '12 YEARS', [20, 940, 100, 960]),
               self.candidate(3, '2', [20, 20, 200, 150])]
        kept, excluded = filter_candidates(raw, image_size=(800, 1000))
        self.assertEqual([c['id'] for c in kept], ['c1', 'c2', 'c3'])
        self.assertEqual(excluded[0]['reason'], 'page_number')

    def test_watermark_phrases_removed_without_deleting_dialogue_words(self):
        excluded_texts = ['MANGA', 'do not mirror', 'DO  NOT\nMIRROR',
                          'stream.com', 'DONOTMIRROR', '.com',
                          'MANGA DO NOT MIRROR', 'STREAM.com DONOTMIRROR',
                          'STREAM', 'STREAM.om DONOT MIRROR']
        preserved = ['I READ MANGA.', 'DO NOT DROP IT!', 'LOOK IN THE MIRROR.',
                     'MANGANESE', 'FOLLOW THE STREAM.', '...', '?!']
        raw = [self.candidate(i, text) for i, text in enumerate(excluded_texts + preserved)]
        kept, removed = filter_candidates(raw)
        self.assertEqual([c['text'] for c in kept], preserved)
        self.assertEqual(len(removed), len(excluded_texts))
        self.assertTrue(all(e['reason'] == 'watermark_phrase' for e in removed))

    def test_saved_split_used_without_training_targets(self):
        split = json.loads((ROOT / 'outputs/split.json').read_text(encoding='utf-8'))
        _, rows = select_rows(ROOT, 'val')
        self.assertEqual([r['sequence_id'] for r in rows], split['val'])
        self.assertTrue(all(set(r) == {'sequence_id', 'images'} for r in rows))
        with self.assertRaises(ValueError):
            select_rows(ROOT, 'val', limit=0)


if __name__ == '__main__':
    unittest.main()
