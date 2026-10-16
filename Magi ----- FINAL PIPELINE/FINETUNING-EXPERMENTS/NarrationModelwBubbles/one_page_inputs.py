"""Reusable cached one-page inputs and optional trained-classifier inference."""
from pathlib import Path
import json
import gc
import numpy as np
from PIL import Image
import torch

from config import CONFIG, BUBBLE_SETTINGS, ROOT, fingerprint
from bubble_features import (sha256, model_files, save_json, replace_visuals,
                             original_page_features, semantic_features)


def prepare_page(page_path, raw_magi_cache=None):
    page_path = Path(page_path)
    page_hash = sha256(page_path)
    identity = dict(page_sha256=page_hash, magi_files=model_files(Path(CONFIG['magi_model'])),
                    helper_sha256=sha256(ROOT / 'one_page_inputs.py'))
    folder = CONFIG['cache_dir'] / 'one_page' / (page_path.parent.name + '_' + page_path.stem + '_' + page_hash[:12])
    folder.mkdir(parents=True, exist_ok=True)
    if raw_magi_cache is not None:
        raw_path = Path(raw_magi_cache)
        payload = json.loads(raw_path.read_text(encoding='utf-8'))
        if payload.get('data_hash') != fingerprint(payload['data']):
            raise ValueError('Corrupt raw Magi cache')
        manifest_path = raw_path.parent / 'manifest.json'
        metadata = (json.loads(manifest_path.read_text(encoding='utf-8'))['inputs']['pages']
                    if manifest_path.exists() else payload.get('settings', {}).get('pages', []))
        matching = [i for i, p in enumerate(metadata) if p['sha256'] == page_hash and p['name'] == page_path.name]
        if len(matching) != 1:
            raise ValueError('Raw Magi cache has no unique matching page hash/name; set RAW_MAGI_CACHE=None to infer.')
        result = payload['data'][matching[0]]
    else:
        raw_path = folder / 'magi_raw.json'
        raw_key = fingerprint(identity)
        payload = json.loads(raw_path.read_text(encoding='utf-8')) if raw_path.exists() else None
        if payload is not None and payload.get('signature') == raw_key and payload.get('data_hash') == fingerprint(payload['data']):
            result = payload['data']
        else:
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(CONFIG['magi_model'], trust_remote_code=True, local_files_only=True)
            cfg.detection_model_config.use_pretrained_backbone = False
            cfg.disable_ocr = False
            model = AutoModel.from_pretrained(CONFIG['magi_model'], config=cfg,
                trust_remote_code=True, local_files_only=True).to(BUBBLE_SETTINGS['device']).eval()
            try:
                with Image.open(page_path) as image:
                    image = np.asarray(image.convert('L').convert('RGB'))
                with torch.inference_mode():
                    result = model.do_chapter_wide_prediction([image], {'images': [], 'names': []}, do_ocr=True)[0]
                def plain(value):
                    if isinstance(value, dict):
                        return {k: plain(v) for k, v in value.items()}
                    if isinstance(value, (list, tuple)):
                        return [plain(v) for v in value]
                    return value.tolist() if hasattr(value, 'tolist') else value
                result = plain(result)
                save_json(raw_path, dict(signature=raw_key, data=result, data_hash=fingerprint(result)))
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
    with Image.open(page_path) as image:
        width, height = image.size
    if len(result['texts']) != len(result['ocr']):
        raise ValueError('Magi boxes/OCR count mismatch')
    boxes = [dict(box_id=f'PG1:T{i+1}', page_index=0, text_index=i, bbox=box, ocr=text,
                  label=-100, gt_speaker=None) for i, (box, text) in enumerate(zip(result['texts'], result['ocr']))]
    sequence = dict(sequence_id=folder.name,
        pages=[dict(path=str(page_path), sha256=page_hash, width=width, height=height)], boxes=boxes)
    from ocr_selection import select_sequences
    selected, decisions = select_sequences([sequence], CONFIG['cache_dir'] / 'ocr/one_page')
    sequence = selected[0]
    boxes = sequence['boxes']
    contract = dict(identity=identity, sequence=sequence, settings=BUBBLE_SETTINGS,
        detector_sha256=sha256(BUBBLE_SETTINGS['bubble_model']),
        bubble_helper_sha256=sha256(ROOT / 'bubble_features.py'),
        semantic_files=model_files(Path(CONFIG['deberta_model'])), max_tokens=CONFIG['max_tokens'],
        packages={'torch': str(torch.__version__), 'transformers': __import__('transformers').__version__,
                  'ultralytics': __import__('ultralytics').__version__})
    key = fingerprint(contract)
    cache = folder / f'inputs_{key[:16]}.pt'
    if cache.exists():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved.get('signature') == key:
            print('CACHE one-page inputs:', cache)
            return saved, folder
    original, tails = original_page_features(sequence, CONFIG['magi_model'], BUBBLE_SETTINGS['device'])
    vectors, summary = replace_visuals([sequence], BUBBLE_SETTINGS,
        {sequence['sequence_id']: original}, CONFIG['cache_dir'] / 'bubbles/one_page')
    semantic = semantic_features(sequence, CONFIG['deberta_model'], CONFIG['max_tokens'],
                                 CONFIG['semantic_batch_size'], BUBBLE_SETTINGS['device'])
    ids = [b['box_id'] for b in boxes]
    n = len(ids)
    geometry = []
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        geometry.append([(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height])
    batch = dict(text=torch.stack([semantic[i] for i in ids])[None] if n else torch.empty(1, 0, 768),
        visual=torch.stack([vectors[sequence['sequence_id']][i] for i in ids])[None] if n else torch.empty(1, 0, 256),
        tail=torch.stack([tails[i] for i in ids])[None] if n else torch.empty(1, 0, 2),
        geometry=torch.tensor(geometry, dtype=torch.float).reshape(1, n, 4),
        page=torch.zeros((1, n), dtype=torch.long), attention_mask=torch.ones((1, n), dtype=torch.bool))
    saved = dict(signature=key, contract=contract, sequence=sequence, batch=batch, summary=summary)
    torch.save(saved, cache)
    save_json(folder / 'inputs_metadata.json', dict(signature=key, cache=str(cache), contract=contract, summary=summary))
    return saved, folder


def classify_page(inputs, checkpoint_path, threshold=.5):
    from model import BoxClassifier
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    contract = ckpt.get('feature_contract', {})
    if contract.get('name') != 'NarrationModelwBubbles':
        raise ValueError('Use a trained NarrationModelwBubbles checkpoint, not the original/DINO classifier.')
    expected = dict(contract['settings'])
    actual = dict(BUBBLE_SETTINGS)
    expected.pop('device', None)
    actual.pop('device', None)
    if expected != actual or contract['helper_sha256'] != sha256(ROOT / 'bubble_features.py'):
        raise ValueError('Checkpoint bubble feature contract differs; restore matching settings/helper or retrain.')
    boxes, batch = inputs['sequence']['boxes'], inputs['batch']
    if len(boxes) > ckpt['max_positions']:
        raise ValueError('Page exceeds trained position capacity; no boxes were truncated.')
    if (contract['bubble_model_sha256'] != sha256(BUBBLE_SETTINGS['bubble_model']) or
        contract['magi_files'] != model_files(Path(CONFIG['magi_model'])) or
        contract['deberta_files'] != model_files(Path(CONFIG['deberta_model'])) or
        contract['max_tokens'] != CONFIG['max_tokens']):
        raise ValueError('Checkpoint encoder/detector/preprocessing identity differs from the current inputs.')
    if batch['visual'].shape[-1] != ckpt['visual_dim']:
        raise ValueError('Visual dimension mismatch')
    if not boxes:
        return []
    model = BoxClassifier(ckpt['visual_dim'], ckpt['max_positions'], **ckpt['model_args']).eval()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    with torch.inference_mode():
        logits = model(batch)[0]
        if logits.shape != (len(boxes), 2) or not torch.isfinite(logits).all():
            raise ValueError('Invalid classifier output')
        probabilities = logits.softmax(-1)
    return [dict(box_id=b['box_id'], bbox=b['bbox'], text=b['ocr'],
        p_spoken=float(p[0]), p_narrated=float(p[1]),
        text_role='NARRATED' if float(p[1]) >= threshold else 'SPOKEN') for b,p in zip(boxes, probabilities)]
