"""Run saved-split OCR, sound-word filtering, cleanup, and geometry joining.

No reference text, labels, or manual review decisions enter this pipeline.
Sound filtering uses the explicit common-word blacklist, without visual classification.
"""
import json
import math
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from PIL import Image

from data_utils import load_jsonl
from ocr_joining import join_ocr_candidates
from ocr_utils import atomic_write, discover_sequences, sha256

SOUND_EFFECTS_PATH = Path(__file__).resolve().parents[1] / 'datasets/rules/common_sound_effects.json'


def normalize_sound_word(text):
    text = ' '.join(unicodedata.normalize('NFKC', text).upper().split())
    return re.sub(r'^[\W_]+|[\W_]+$', '', text)


def load_sound_words(path=SOUND_EFFECTS_PATH):
    return {normalize_sound_word(word) for word in json.loads(Path(path).read_text(encoding='utf-8'))['words']}


def load_watermark_phrases(path=SOUND_EFFECTS_PATH):
    return {normalize_sound_word(phrase) for phrase in
            json.loads(Path(path).read_text(encoding='utf-8')).get('watermark_phrases', [])}


def is_watermark_fragment(text, phrases):
    """Match complete watermark phrases or combinations, never dialogue substrings."""
    if not phrases:
        return False
    alternatives = '|'.join(re.escape(p) for p in sorted(phrases, key=len, reverse=True) if p)
    return bool(re.fullmatch(rf'(?:{alternatives})(?:\s+(?:{alternatives}))*', normalize_sound_word(text)))


def select_rows(root, split='val', input_root=None, limit=None):
    root = Path(root).resolve()
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or None')
    source = Path(input_root).resolve() if input_root else root / 'TaskData'
    rows = discover_sequences(source)
    if input_root is None:
        saved = json.loads((root / 'outputs/split.json').read_text(encoding='utf-8'))
        train, val = saved['train'], saved['val']
        if len(set(train + val)) != len(train + val):
            raise ValueError('Saved train/validation split overlaps or duplicates IDs')
        manifest = json.loads((source / 'sequences.json').read_text(encoding='utf-8'))
        dev = {r['sequence_id'] for r in manifest if r['split'] == 'development'}
        if set(train + val) != dev:
            raise ValueError('Saved split does not cover the development manifest')
        ids = {'train': train, 'val': val,
               'test': [r['sequence_id'] for r in manifest if r['split'] == 'test'],
               'all': [r['sequence_id'] for r in manifest]}
        if split not in ids:
            raise ValueError('split must be train, val, test, or all')
        by_id = {r['sequence_id']: r for r in rows}
        rows = [by_id[sid] for sid in ids[split]]
    return source, rows[:limit] if limit is not None else rows


# Ordinary English typography is allowed alongside ASCII letters and punctuation.
ENGLISH_PUNCTUATION = set("‘’“”–—…")
PAGE_NUMBER_MARGIN = 0.08  # Top/bottom 8% of the original page.


def has_non_english_characters(text):
    """Reject the whole fragment, including mixed English/Japanese text."""
    normalized = unicodedata.normalize('NFKC', text)
    return any(ord(char) > 127 and char not in ENGLISH_PUNCTUATION
               and not char.isspace() for char in normalized)


def is_sound_effect(text, sound_words):
    """Match listed whole words, repeated sound words, or explicit SFX captions."""
    normalized = normalize_sound_word(text)
    if normalized.startswith('SFX:'):
        return True
    if normalized in sound_words:
        return True
    words = [normalize_sound_word(word) for word in normalized.split()]
    return bool(words) and all(word in sound_words for word in words)


def is_page_number(candidate, image_size):
    """Small standalone integer near the top/bottom; preserve dialogue numbers."""
    if image_size is None:
        return False  # Position cannot be checked without original dimensions.
    text = unicodedata.normalize('NFKC', candidate['text']).strip()
    if not re.fullmatch(r'[0-9]{1,4}', text):
        return False
    width, height = image_size
    x0, y0, x1, y1 = candidate['box']
    center_y = (y0 + y1) / 2 / height
    near_margin = center_y <= PAGE_NUMBER_MARGIN or center_y >= 1 - PAGE_NUMBER_MARGIN
    small_box = (x1 - x0) <= width * 0.08 and (y1 - y0) <= height * 0.05
    return near_margin and small_box


def exclusion_reason(candidate, min_score, sound_words, watermark_phrases, image_size):
    """One readable check per rule; the first matching rule is recorded in the audit."""
    text, box, score = candidate['text'], candidate['box'], candidate['score']
    if not isinstance(text, str):
        raise ValueError('Candidate text must be a string')
    if not text.strip():
        return 'blank_text'
    valid_box = (isinstance(box, list) and len(box) == 4 and
                 all(type(v) in (int, float) and math.isfinite(v) for v in box))
    if not valid_box or box[2] <= box[0] or box[3] <= box[1]:
        return 'invalid_box'
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return 'invalid_score'
    if score < min_score:
        return 'below_min_score'
    if has_non_english_characters(text):
        return 'non_english_fragment'
    if is_watermark_fragment(text, watermark_phrases):
        return 'watermark_phrase'
    if is_sound_effect(text, sound_words):
        return 'sound_effect_word'
    if is_page_number(candidate, image_size):
        return 'page_number'
    return None


def filter_candidates(candidates, min_score=0.0, sound_words=None,
                      watermark_phrases=None, image_size=None):
    """Apply every exclusion BEFORE joining; retain original text for kept fragments."""
    if not 0 <= min_score <= 1:
        raise ValueError('min_score must be between zero and one')
    if sound_words is None:
        sound_words = load_sound_words()
    if watermark_phrases is None:
        watermark_phrases = load_watermark_phrases()
    valid, excluded, seen = [], [], set()
    for candidate in candidates:
        cid = candidate['id']
        if not isinstance(cid, str) or cid in seen:
            raise ValueError('Candidate IDs must be unique strings within a page')
        seen.add(cid)
        reason = exclusion_reason(candidate, min_score, sound_words, watermark_phrases, image_size)
        if reason is None:
            valid.append(candidate)
        else:
            excluded.append(dict(candidate=candidate, reason=reason))

    # For identical text at identical coordinates, retain the strongest detection.
    winners = {}
    for candidate in valid:
        key = (candidate['text'].strip(), tuple(candidate['box']))
        if key not in winners or candidate['score'] > winners[key]['score']:
            winners[key] = candidate
    kept = []
    for candidate in valid:
        key = (candidate['text'].strip(), tuple(candidate['box']))
        if candidate is winners[key]:
            kept.append(candidate)
        else:
            excluded.append(dict(candidate=candidate, reason='exact_duplicate'))
    return kept, excluded


def load_cache(path, rows):
    path = Path(path)
    if not path.is_file():
        return {}, None
    cache = {}
    for record in load_jsonl(path):
        sid = record['sequence_id']
        if sid in cache or not isinstance(record.get('pages'), list) or len(record['pages']) != 3:
            raise ValueError(f'Invalid or duplicate raw OCR sequence: {sid}')
        cache[sid] = record
    metadata_path = path.with_suffix('.metadata.json')
    provenance = dict(path=str(path.resolve()), sha256=sha256(path),
                      provenance_status='legacy_cache_without_image_or_runtime_provenance')
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('output_sha256') != provenance['sha256']:
            raise ValueError('Raw cache was modified after extraction')
        for row in rows:
            sid = row['sequence_id']
            if sid in cache and metadata['sequences'].get(sid) != [sha256(p) for p in row['images']]:
                raise ValueError(f'{sid}: cached OCR belongs to different images')
        provenance.update(provenance_status='image_hashes_verified',
                          metadata_sha256=sha256(metadata_path), runtime=metadata['runtime'])
    return cache, provenance


def run_pipeline(root, *, split='val', input_root=None, raw_cache=None, output_dir=None,
                 ocr_python=None, limit=None, min_score=0.0, max_gap=1.25, center_tolerance=1.5,
                 sound_effects_path=SOUND_EFFECTS_PATH):
    root = Path(root).resolve()
    source, rows = select_rows(root, split, input_root, limit)
    output = Path(output_dir).resolve() if output_dir else root / 'outputs/text_extraction' / split
    raw_cache = Path(raw_cache) if raw_cache else (root / 'outputs/ocr/paddleocr/candidates.jsonl'
                                                  if input_root is None else output / 'raw_candidates.jsonl')
    final = output / 'candidates.jsonl'
    if final.resolve() == raw_cache.resolve() or final.is_relative_to(source):
        raise ValueError('Final output must be separate from raw OCR and input data')
    cache, provenance = load_cache(raw_cache, rows)
    sources = [provenance] if provenance else []
    missing = [r for r in rows if r['sequence_id'] not in cache]
    if missing:
        # A separate new cache protects the supplied legacy candidates file.
        fresh_path = output / 'raw_candidates.jsonl'
        fresh, fresh_provenance = load_cache(fresh_path, rows)
        still_missing = [r for r in missing if r['sequence_id'] not in fresh]
        if still_missing:
            interpreter = Path(ocr_python or os.environ.get('MANGA_OCR_PYTHON',
                               str(root.parent / '.venvOCR/Scripts/python.exe')))
            if not interpreter.is_file():
                raise ValueError('Set ocr_python or MANGA_OCR_PYTHON to the OCR interpreter')
            command = [str(interpreter), str(root / 'scripts/build_ocr_cache.py'),
                       '--input', str(source), '--output', str(fresh_path)]
            for row in still_missing:
                command.extend(['--sequence-id', row['sequence_id']])
            subprocess.run(command, check=True)
            fresh, fresh_provenance = load_cache(fresh_path, rows)
        cache.update(fresh)
        if fresh_provenance and fresh_provenance not in sources:
            sources.append(fresh_provenance)
    sound_words = load_sound_words(sound_effects_path)
    watermark_phrases = load_watermark_phrases(sound_effects_path)
    records, audit = [], []
    for row in rows:
        pages = []
        for index, raw in enumerate(cache[row['sequence_id']]['pages']):
            # Read original dimensions for page-number detection; OCR stays on GPU.
            with Image.open(row['images'][index]) as image:
                image_size = image.size
            kept, excluded = filter_candidates(
                raw, min_score, sound_words, watermark_phrases, image_size=image_size)
            # Only clean fragments are allowed to enter the geometry joiner.
            groups = join_ocr_candidates(kept, max_gap=max_gap, center_tolerance=center_tolerance)
            scores = {c['id']: c['score'] for c in kept}
            for group in groups:
                group['score'] = sum(scores[cid] for cid in group['candidate_ids']) / len(group['candidate_ids'])
            pages.append(groups)
            audit.append(dict(sequence_id=row['sequence_id'], page_index=index,
                              raw_count=len(raw), kept_count=len(kept), joined_count=len(groups),
                              excluded=excluded))
        records.append(dict(sequence_id=row['sequence_id'], pages=pages))
    atomic_write(final, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    audit_path = output / 'filter_audit.jsonl'
    atomic_write(audit_path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in audit))
    metadata = dict(split=split if input_root is None else 'unseen',
                    sequence_ids=[r['sequence_id'] for r in rows], raw_sources=sources,
                    images={r['sequence_id']: [sha256(p) for p in r['images']] for r in rows},
                    config=dict(min_score=min_score, max_gap=max_gap, center_tolerance=center_tolerance,
                                english_only=True, page_number_margin=PAGE_NUMBER_MARGIN,
                                page_number_max_width=0.08, page_number_max_height=0.05),
                    code={name: sha256(root / 'src' / name) for name in
                          ('text_extraction_pipeline.py', 'ocr_joining.py', 'ocr_utils.py')},
                    output_sha256=sha256(final), audit_sha256=sha256(audit_path),
                    sound_effects_sha256=sha256(sound_effects_path), sound_effect_words=sorted(sound_words),
                    watermark_phrases=sorted(watermark_phrases),
                    filtering_scope='Before joining: non-English fragments, watermarks, listed/repeated sounds, SFX captions, margin page numbers, and OCR cleanup; no visual classification')
    if input_root is None:
        metadata['split_sha256'] = sha256(root / 'outputs/split.json')
    atomic_write(final.with_suffix('.metadata.json'), json.dumps(metadata, indent=2) + '\n')
    summary = dict(sequences=len(records), pages=len(audit),
                   raw_candidates=sum(p['raw_count'] for p in audit),
                   excluded_candidates=sum(len(p['excluded']) for p in audit),
                   joined_candidates=sum(p['joined_count'] for p in audit), output=str(final))
    print(json.dumps(summary, indent=2))
    return records, summary
