"""Protect ordered label assignment and sequence-wide identity permutations."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vl_prompts import (ordered_groups, build_speaker_prompt, build_speaker_target,
                        parse_speaker_output, permute_speaker_labels)


class SpeakerLinesTests(unittest.TestCase):
    def setUp(self):
        # Deliberately nonalphabetical: p0_g10 must not move ahead of p0_g2.
        self.groups = [dict(group_id=gid, text='Hello\nthere', box=[1, 2, 3, 4])
                       for gid in ['p0_g2', 'p0_g10', 'p1_g0']]

    def test_target_and_parse_preserve_order(self):
        labelled = [dict(g, speaker=s) for g, s in zip(self.groups, ['char2', 'UNKNOWN', 'char1'])]
        target = build_speaker_target(labelled)
        self.assertEqual(target, 'char2\nUNKNOWN\nchar1')
        self.assertEqual(list(parse_speaker_output(target, self.groups).items()),
                         [('p0_g2', 'char2'), ('p0_g10', 'UNKNOWN'), ('p1_g0', 'char1')])

    def test_prompt_has_text_boxes_but_no_reference_speakers(self):
        prompt = build_speaker_prompt([dict(g, speaker='char987') for g in self.groups])
        self.assertIn('text="Hello\\nthere" box=[1, 2, 3, 4]', prompt)
        self.assertNotIn('char987', prompt)
        self.assertLess(prompt.index('p0_g2:'), prompt.index('p0_g10:'))

    def test_strict_parser_rejects_shifted_or_malformed_responses(self):
        for text in ['char1\nchar2', 'char1\nchar2\nchar3\nchar4',
                     'char1\nchar0\nchar2', 'char1\nchar01\nchar2',
                     'char1\nExplanation\nchar2', '{"p0_g2":"char1"}',
                     '```\nchar1\nchar2\nchar3\n```']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_speaker_output(text, self.groups)

    def test_permutation_keeps_identity_partitions_and_special_classes(self):
        original = ['char1', 'char1', 'char2', 'UNKNOWN', 'char3', 'NARRATION', 'char2']
        rng = random.Random(42)
        presentations = []
        for _ in range(20):
            result = permute_speaker_labels(original, rng)
            presentations.append(result)
            self.assertEqual(result[3], 'UNKNOWN')
            self.assertEqual(result[5], 'NARRATION')
            for i in range(len(original)):
                for j in range(len(original)):
                    self.assertEqual(original[i] == original[j], result[i] == result[j])
        self.assertTrue(any(result != original for result in presentations))
        replay = random.Random(42)
        self.assertEqual(presentations, [permute_speaker_labels(original, replay) for _ in range(20)])

    def test_page_order_and_duplicate_ids(self):
        pages = [dict(page_index=i, groups=[]) for i in range(3)]
        pages[0]['groups'] = [dict(id='p0_g2', text='Hi', box=[1, 2, 3, 4])]
        self.assertEqual(ordered_groups(pages)[0]['group_id'], 'p0_g2')
        with self.assertRaises(ValueError):
            ordered_groups(pages[::-1])
        with self.assertRaises(ValueError):
            parse_speaker_output('char1\nchar2', [self.groups[0], self.groups[0]])


if __name__ == '__main__':
    unittest.main()
