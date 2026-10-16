"""Local exclusion hints and application of explicit visual review decisions."""
import json
import re
import unicodedata
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[1] / "datasets/rules/text_exclusion_lexicon.json"


def normalize_hint(text):
    """Ignore case/spacing/terminal punctuation, preserving punctuation-only text."""
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    trimmed = text.strip(".!?,;: ")
    return trimmed or text


def load_lexicon(path=LEXICON_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def filter_confirmed_sound_effects(candidates, reviews):
    """Remove reviewed SFX before grouping; reject stale/contradictory reviews.

    Reviews must already be scoped to one sequence/page. Hints, reference
    mismatches, and unreviewed rows never trigger removal. Raw OCR is unchanged.
    """
    by_id = {c["id"]: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Candidate IDs must be unique within a page")
    removed, decisions = {}, {}
    for row in reviews:
        if row.get("keep") is None:
            continue
        if type(row["keep"]) is not bool:
            raise ValueError("Review keep must be true, false, or null")
        ids = row.get("candidate_ids", [])
        if not ids or len(set(ids)) != len(ids) or any(cid not in by_id for cid in ids):
            raise ValueError("Review contains missing, repeated, or unknown candidate IDs")
        text = " ".join(by_id[cid]["text"].strip() for cid in ids)
        if text != row["text"]:
            raise ValueError("Reviewed text differs from OCR; review the current cache again")
        drop = row["keep"] is False and row.get("exclusion_category") == "visual_sound_effect"
        for cid in ids:
            if cid in decisions and decisions[cid] != drop:
                raise ValueError("Conflicting sound-effect review decisions")
            decisions[cid] = drop
            if drop:
                removed[cid] = dict(**by_id[cid], exclusion_category="visual_sound_effect")
    return ([c for c in candidates if c["id"] not in removed], list(removed.values()))


def exclusion_hints(text, lexicon=None):
    """Match whole regions only; visual context must confirm each suggestion.

    No fuzzy/substring word filtering: 'BAM' must not flag 'BAMBOO'. Numeric
    balloons, vocalisations, and punctuation-only balloons remain valid text.
    """
    lexicon = load_lexicon() if lexicon is None else lexicon
    key = normalize_hint(text)
    hints = [dict(category=entry["category"], term=entry["term"],
                  requires_visual_confirmation=True)
             for entry in lexicon["entries"]
             if key in {normalize_hint(v) for v in [entry["term"], *entry.get("variants", [])]}]
    if re.fullmatch(r"\d+", key):
        hints.append(dict(category="possible_page_number_or_object_text", term=text,
                          requires_visual_confirmation=True))
    return hints
