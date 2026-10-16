"""The exact Magi + Paddle validity/selection rules from pipelineFULL_INFERENCE."""
from pathlib import Path
import copy
import gc
import json
import os
import unicodedata
import numpy as np
from PIL import Image
from config import CONFIG, ROOT, fingerprint, save_json
from bubble_features import sha256, model_files


def select_sequences(sequences, output_root=None, force=False):
    """Run/cache Paddle per original Magi box; preserve labels, boxes, and text order."""
    import torch  # Must precede Paddle in Godfather.
    import paddle
    from paddleocr import PaddleOCR
    from importlib.metadata import version
    from config import (PADDLE_SETTINGS, PADDLE_LOCAL_SETTINGS,
                        PADDLE_DET_PATH, PADDLE_REC_PATH, PREFERRED_OCR)
    output_root = Path(output_root or CONFIG['cache_dir'] / 'ocr')
    identity = dict(paddle=PADDLE_SETTINGS, local_models=PADDLE_LOCAL_SETTINGS,
        detector_files=model_files(PADDLE_DET_PATH), recognizer_files=model_files(PADDLE_REC_PATH),
        prefer=PREFERRED_OCR, check='unicode_text_validity_v1', code_sha256=sha256(Path(__file__)),
        packages={k:version(k) for k in ['paddlepaddle-gpu', 'paddleocr', 'paddlex', 'numpy', 'pillow']})
    engine, output, all_decisions = None, [], {}
    try:
        for original in sequences:
            seq = copy.deepcopy(original)
            sid = seq['sequence_id']
            decisions = []
            for page_index, page in enumerate(seq['pages']):
                if sha256(page['path']) != page['sha256']:
                    raise ValueError('OCR source page changed')
                boxes = [b for b in seq['boxes'] if b['page_index'] == page_index]
                # Always compare against original Magi, including when refreshing a selected cache.
                inputs = [dict(box_id=b['box_id'], bbox=b['bbox'], magi=b.get('magi_ocr', b['ocr'])) for b in boxes]
                key = fingerprint(dict(page=page, inputs=inputs, identity=identity))
                path = output_root / sid / f'page_{page_index+1:02d}_{key[:16]}.json'
                payload = json.loads(path.read_text(encoding='utf-8')) if path.exists() and not force else None
                if payload is not None and (payload.get('signature') != key or payload.get('data_hash') != fingerprint(payload['data'])):
                    payload = None
                if payload is None:
                    if engine is None:
                        if not PADDLE_DET_PATH.is_dir() or not PADDLE_REC_PATH.is_dir():
                            raise FileNotFoundError('Set local Paddle detector/recognizer paths in config.py')
                        if not paddle.is_compiled_with_cuda():
                            raise RuntimeError('Godfather Paddle CUDA runtime is required')
                        engine = PaddleOCR(**PADDLE_SETTINGS, **PADDLE_LOCAL_SETTINGS)
                    with Image.open(page['path']) as image:
                        image = np.asarray(image.convert('L').convert('RGB'))
                    records = []
                    for text_index, (box, item) in enumerate(zip(boxes, inputs)):
                        crop, clipped = crop_box(image, box['bbox'])
                        record = dict(page_index=page_index, text_index=box['text_index'], text_id=f"T{box['text_index']+1}",
                            box=box['bbox'], crop_box=clipped, text='', lines=[], status='invalid_crop', error=None)
                        if crop is not None:
                            try:
                                for prediction in engine.predict(input=crop):
                                    texts, scores = prediction['rec_texts'], prediction['rec_scores']
                                    if len(texts) != len(scores):
                                        raise ValueError('Paddle text/score count mismatch')
                                    record['lines'].extend(dict(text=str(t), confidence=float(s)) for t,s in zip(texts, scores))
                                record['text'] = ' '.join(line['text'] for line in record['lines'])
                                record['status'] = 'text' if record['text'].strip() else 'empty'
                            except Exception as error:
                                record.update(text='', lines=[], status='error', error=str(error))
                        decision = select_ocr(record['text'], item['magi'], PREFERRED_OCR)
                        decision.update(box_id=box['box_id'], page_index=page_index,
                            text_index=box['text_index'], box=box['bbox'], paddle_record=record)
                        records.append(decision)
                    payload = dict(signature=key, identity=identity, data=records, data_hash=fingerprint(records))
                    save_json(path, payload)
                    status = 'RUN'
                else:
                    status = 'CACHE'
                if len(payload['data']) != len(boxes):
                    raise ValueError('Selected OCR count mismatch')
                for box, decision in zip(boxes, payload['data']):
                    if box['box_id'] != decision['box_id'] or box['bbox'] != decision['box']:
                        raise ValueError('Selected OCR text identity mismatch')
                    box['magi_ocr'] = box.get('magi_ocr', box['ocr'])
                    box['ocr'] = decision['selected_text']
                    box['ocr_selection'] = decision
                decisions.extend(payload['data'])
                print(f'{status} selected OCR {sid} page {page_index+1}: {len(boxes)} boxes', flush=True)
            seq['ocr_contract'] = identity
            output.append(seq)
            all_decisions[sid] = decisions
    finally:
        del engine
        gc.collect()
        paddle.device.cuda.empty_cache()
    return output, all_decisions

def crop_box(image, box):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None, None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = values
    clipped = [int(np.clip(np.floor(x1), 0, width)), int(np.clip(np.floor(y1), 0, height)),
               int(np.clip(np.ceil(x2), 0, width)), int(np.clip(np.ceil(y2), 0, height))]
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        return None, clipped
    return np.ascontiguousarray(image[y1:y2, x1:x2]), clipped


def text_validity(text):
    if text is None:
        return dict(passes=False, reason='empty', text='')
    if not isinstance(text, str):
        return dict(passes=False, reason='not_a_string', text='')
    normalized = ' '.join(text.split())
    if not normalized:
        reason = 'empty'
    elif '\ufffd' in text or any(unicodedata.category(c) in {'Cc', 'Cs'} and c not in '\n\r\t' for c in text):
        reason = 'invalid_characters'
    elif not any(c.isalnum() for c in normalized):
        reason = 'no_letters_or_digits'
    else:
        reason = 'valid_text'
    return dict(passes=reason == 'valid_text', reason=reason, text=normalized)


def select_ocr(paddle_text, magi_text, prefer='paddle'):
    assert prefer in {'paddle', 'magi'}, 'Choose paddle or magi'
    checks = {'paddle': text_validity(paddle_text), 'magi': text_validity(magi_text)}
    passing = [source for source, check in checks.items() if check['passes']]
    nonempty = [source for source, check in checks.items() if check['text']]
    if len(passing) == 1:
        source, reason = passing[0], 'only_valid_output'
    elif len(passing) == 2:
        source, reason = prefer, 'both_valid_preferred'
    elif len(nonempty) == 1:
        source, reason = nonempty[0], 'only_nonempty_needs_review'
    else:
        source = None
        reason = 'both_empty' if all(c['reason'] == 'empty' for c in checks.values()) else 'both_invalid'
    return dict(paddle_text=paddle_text, magi_text=magi_text, validity=checks,
                selected_text=checks[source]['text'] if source else '', selected_source=source,
                reason=reason, needs_review=not bool(passing),
                disagreement=len(passing) == 2 and checks['paddle']['text'].casefold() != checks['magi']['text'].casefold())


