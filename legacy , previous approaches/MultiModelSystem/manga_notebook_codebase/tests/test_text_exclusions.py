import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from text_exclusions import exclusion_hints, filter_confirmed_sound_effects
from build_exclusion_review import build_review


class TextExclusionTests(unittest.TestCase):
    def test_confirmed_sfx_is_removed_before_joining_without_deleting_dialogue(self):
        from ocr_joining import join_ocr_candidates
        candidates = [dict(id="c0", text="BANG!", box=[0, 0, 80, 10]),
                      dict(id="c1", text="BANG!", box=[0, 15, 80, 25]),
                      dict(id="c2", text="...", box=[0, 30, 80, 40])]
        review = dict(candidate_ids=["c0"], text="BANG!", keep=False,
                      exclusion_category="visual_sound_effect")
        kept, removed = filter_confirmed_sound_effects(candidates, [review])
        self.assertEqual([c["id"] for c in removed], ["c0"])
        self.assertEqual(join_ocr_candidates(kept)[0]["text"], "BANG! ...")
        self.assertEqual(len(candidates), 3)
        kept, removed = filter_confirmed_sound_effects(candidates, [dict(review, keep=None)])
        self.assertEqual(kept, candidates)
        self.assertEqual(removed, [])

    def test_stale_and_conflicting_reviews_fail(self):
        candidates = [dict(id="c0", text="BANG!", box=[0, 0, 80, 10])]
        review = dict(candidate_ids=["c0"], text="BANG!", keep=False,
                      exclusion_category="visual_sound_effect")
        for reviews in [[dict(review, text="old text")],
                        [dict(review, candidate_ids=["missing"])],
                        [review, dict(review, keep=True)]]:
            with self.assertRaises(ValueError):
                filter_confirmed_sound_effects(candidates, reviews)

    def test_protect_speech_and_punctuation(self):
        for text in ["...", "?!", "!", "YAY!", "HEH", "GUH GUH", "AH!",
                     "CAN'T SEE ONE...", "BAMBOO", "I heard a bang."]:
            self.assertEqual(exclusion_hints(text), [], text)

    def test_hints_require_context(self):
        for text in [" thud! ", "SFX: GULP", "STREAM.com DO NOT MIRROR", "42"]:
            hints = exclusion_hints(text)
            self.assertTrue(hints, text)
            self.assertTrue(all(h["requires_visual_confirmation"] for h in hints))

    def test_inventory_keeps_unmatched_dialogue_unlabeled_and_protects_reviews(self):
        token = uuid.uuid4().hex
        source = ROOT / "outputs" / f"test_source_{token}.jsonl"
        output = ROOT / "outputs" / f"test_review_{token}.jsonl"
        try:
            source.write_text(json.dumps(dict(sequence_id="s", page_index=0, groups=[
                dict(id="g0", candidate_ids=["c0"], text="YAY!", box=[0, 0, 1, 1], target_index=None),
                dict(id="g1", candidate_ids=["c1"], text="Hi", box=[0, 0, 1, 1], target_index=0)
            ])) + "\n", encoding="utf-8")
            self.assertEqual(build_review(source, output), 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["text"], "YAY!")
            self.assertIsNone(row["keep"])
            self.assertEqual(row["hints"], [])
            with self.assertRaises(FileExistsError):
                build_review(source, output)
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
