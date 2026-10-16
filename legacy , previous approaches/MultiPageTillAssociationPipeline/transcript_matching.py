"""Match one bubble's OCR to page GT. No model or file operations."""
import re
from difflib import SequenceMatcher


def normalize_text(text):
    return " ".join(text.upper().split())


def phrase_inside(phrase, text):
    # Word boundaries prevent matching a word inside another word.
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def match_transcript(record, dialogue, minimum_score=80.0, minimum_partial_words=3):
    if minimum_partial_words < 3:
        raise ValueError('Exact partial matching requires at least three words.')
    text = normalize_text(record['text'])
    label = {'bubble_id':record['bubble_id'], 'text':record['text'],
             'gt_index':None, 'gt_text':None, 'speaker':None,
             'match_score':0.0, 'match_type':'unmatched'}
    if not text or not dialogue:
        return label
    gt_texts = [normalize_text(gt['text']) for gt in dialogue]
    candidates = [i for i, gt_text in enumerate(gt_texts) if text == gt_text]
    match_type = 'exact'
    words = re.findall(r"\b\w+(?:['’]\w+)*\b", text)
    if not candidates and len(words) >= minimum_partial_words:
        candidates = [i for i, gt_text in enumerate(gt_texts) if phrase_inside(text, gt_text)]
        match_type = 'exact_partial'
    if candidates:
        speakers = {dialogue[i]['speaker'] for i in candidates}
        if len(speakers) != 1 or None in speakers:
            # Do not let fuzzy matching pick a speaker after an ambiguous exact phrase.
            label.update(match_score=100.0, match_type='ambiguous')
            return label
        index = candidates[0]
        score = 100.0
    else:
        # Same full-string fuzzy comparison as before; no short partial matches.
        scores = [100 * SequenceMatcher(None, text, gt_text).ratio() for gt_text in gt_texts]
        index = max(range(len(scores)), key=lambda i:scores[i])
        score = scores[index]
        label['match_score'] = round(score, 1)
        if score < minimum_score or dialogue[index]['speaker'] is None:
            return label
        match_type = 'fuzzy'
    label.update(gt_index=index, gt_text=dialogue[index]['text'],
                 speaker=dialogue[index]['speaker'], match_score=round(score, 1),
                 match_type=match_type)
    return label
