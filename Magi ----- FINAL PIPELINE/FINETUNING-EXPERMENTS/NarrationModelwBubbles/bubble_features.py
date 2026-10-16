"""Bubble crops -> frozen Magi DETR encoder features. No DINO or Paddle OCR."""
from pathlib import Path
import gc
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def model_files(path):
    return {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns,
            sha256(p) if p.suffix in {'.py', '.json'} else None]
            for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.json', '.safetensors', '.bin'}}


def coverage(text, bubble):
    x1, y1, x2, y2 = text
    bx1, by1, bx2, by2 = bubble
    area = max(0, x2-x1) * max(0, y2-y1)
    overlap = max(0, min(x2, bx2)-max(x1, bx1)) * max(0, min(y2, by2)-max(y1, by1))
    return overlap / area if area else 0.0


def assign_texts(bubbles, texts, minimum_coverage=0.8):
    """One best bubble per nonempty OCR box; deterministic nested-box handling."""
    assigned = {b['bubble_id']: [] for b in bubbles}
    mapping = {}
    for text in texts:
        if not isinstance(text['ocr'], str) or not text['ocr'].strip():
            continue
        candidates = []
        for bubble in bubbles:
            score = coverage(text['bbox'], bubble['box'])
            if score >= minimum_coverage:
                x1, y1, x2, y2 = bubble['box']
                candidates.append((-score, (x2-x1)*(y2-y1), bubble['bubble_id']))
        if candidates:
            bubble_id = min(candidates)[2]
            assigned[bubble_id].append(text['box_id'])
            mapping[text['box_id']] = bubble_id
    records = []
    for bubble in bubbles:
        record = dict(bubble, text_ids=assigned[bubble['bubble_id']])
        record['kept'] = bool(record['text_ids'])
        record['reason'] = 'nonempty_selected_text' if record['kept'] else 'no_assigned_nonempty_selected_text'
        records.append(record)
    return records, mapping


def load_magi(path, device):
    import torch
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    config.detection_model_config.use_pretrained_backbone = False
    config.disable_ocr = True
    config.disable_detections = False
    # Keep crop config: disabling it changes the shape of Magi's matching head.
    # Disabled OCR weights are expected extras; retain explicit checks for missing/mismatched weights.
    from transformers.utils import logging
    verbosity = logging.get_verbosity()
    try:
        logging.set_verbosity_error()
        model, info = AutoModel.from_pretrained(str(path), config=config,
            trust_remote_code=True, local_files_only=True, output_loading_info=True)
    finally:
        logging.set_verbosity(verbosity)
    if info['missing_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'Incompatible local Magi weights: {info}')
    return model.to(device).eval().requires_grad_(False)


def detr_crop_embedding(model, image):
    """Pool spatial encoder tokens over one entire bubble, including border/background.

    This is a NEW 256-D representation, not the original decoder text-query vector.
    Single-image calls avoid batch padding contaminating the spatial average.
    """
    import torch
    array = np.asarray(image.convert('L').convert('RGB'))
    with torch.inference_mode():
        inputs = model.move_to_device(model.processor.preprocess_inputs_for_detection([array]))
        mask = inputs.get('pixel_mask')
        if mask is not None and not bool(mask.all()):
            raise ValueError('Unexpected single-crop padding; implement masked pooling before reuse.')
        output = model._get_detection_transformer_output(**inputs)
        tokens = output.encoder_last_hidden_state
        vector = tokens[0].float().mean(dim=0).cpu()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError('Invalid DETR crop embedding')
    return vector


def extract_bubbles(sequences, settings, output_root, embed=True):
    """Persist every page independently so interrupted runs resume without recomputation."""
    import torch
    from ultralytics import YOLO
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    detector_path = Path(settings['bubble_model'])
    identity = dict(detector_sha256=sha256(detector_path),
        source_code_sha256=sha256(Path(__file__)), settings=settings,
        magi_files=model_files(Path(settings['magi_model'])) if embed else None,
        packages={'torch': str(torch.__version__), 'numpy': np.__version__})
    import transformers, ultralytics, PIL
    identity['packages'].update(transformers=transformers.__version__,
        ultralytics=ultralytics.__version__, pillow=PIL.__version__)
    detector = model = None
    summaries, by_sequence = [], {}
    try:
        for sequence in sequences:
            sid = sequence['sequence_id']
            by_sequence[sid] = {}
            for page_index, page in enumerate(sequence['pages']):
                path = Path(page['path'])
                if sha256(path) != page['sha256']:
                    raise ValueError(f'Source page changed: {path}')
                texts = [b for b in sequence['boxes'] if b['page_index'] == page_index]
                # Labels never enter the crop filtering or embedding contract.
                inputs = [dict(box_id=b['box_id'], bbox=b['bbox'], ocr=b['ocr']) for b in texts]
                key = signature(dict(identity=identity, page=page, texts=inputs, embed=embed))
                folder = output_root / sid / f'page_{page_index+1:02d}' / key[:16]
                cache = folder / 'page.pt'
                saved = torch.load(cache, map_location='cpu', weights_only=True) if cache.exists() else None
                if saved and saved.get('signature') == key:
                    if any(not (folder / b['crop']).is_file() or sha256(folder / b['crop']) != b['crop_sha256']
                           for b in saved['bubbles'] if b['kept']):
                        saved = None
                if saved is None:
                    folder.mkdir(parents=True, exist_ok=True)
                    with Image.open(path) as source:
                        image = source.convert('RGB')
                    if (image.width, image.height) != (page['width'], page['height']):
                        raise ValueError(f'Image dimensions changed: {path}')
                    if detector is None:
                        detector = YOLO(str(detector_path))
                    # Same weights, confidence, image size, rounding and ordering as
                    # 1pageTest/1_bubble_detection/bubble_detection.ipynb.
                    result = detector.predict(source=image, imgsz=settings['bubble_imgsz'],
                        conf=settings['bubble_confidence'], device=settings['device'], verbose=False)[0]
                    bubbles = []
                    for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                        x1, y1, x2, y2 = box
                        bounds = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                                  min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))]
                        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                            bubbles.append(dict(box=bounds, confidence=float(score)))
                    bubbles.sort(key=lambda b: (b['box'][1], -b['box'][0]))
                    for i, bubble in enumerate(bubbles, 1):
                        bubble['bubble_id'] = f'B{i}'
                    records, mapping = assign_texts(bubbles, texts, settings['text_coverage'])
                    vectors = {}
                    preview = image.copy()
                    draw = ImageDraw.Draw(preview)
                    for record in records:
                        bid = record['bubble_id']
                        color = 'green' if record['kept'] else 'red'
                        draw.rectangle(record['box'], outline=color, width=3)
                        draw.text(tuple(record['box'][:2]), bid, fill=color)
                        if not record['kept']:
                            continue
                        crop = image.crop(record['box'])
                        crop_name = f'crops/{bid}.png'
                        (folder / 'crops').mkdir(exist_ok=True)
                        crop.save(folder / crop_name)
                        record.update(crop=crop_name, crop_sha256=sha256(folder / crop_name))
                        if embed:
                            if model is None:
                                model = load_magi(settings['magi_model'], settings['device'])
                            vectors[bid] = detr_crop_embedding(model, crop)
                    preview.save(folder / 'preview.png')
                    saved = dict(signature=key, identity=identity, page=page,
                        bubbles=records, text_to_bubble=mapping, embeddings=vectors,
                        fallback_text_ids=[b['box_id'] for b in texts if b['box_id'] not in mapping])
                    torch.save(saved, cache.with_suffix('.tmp'))
                    cache.with_suffix('.tmp').replace(cache)
                    save_json(folder / 'bubbles.json', {k:v for k,v in saved.items() if k != 'embeddings'})
                    status = 'RUN'
                else:
                    status = 'CACHE'
                by_sequence[sid][page_index] = saved
                row = dict(sequence_id=sid, page_index=page_index, page_file=str(path),
                    candidates=len(saved['bubbles']), kept=sum(b['kept'] for b in saved['bubbles']),
                    rejected=sum(not b['kept'] for b in saved['bubbles']),
                    covered_texts=len(saved['text_to_bubble']), fallback_texts=len(saved['fallback_text_ids']),
                    cache=str(cache), preview=str(folder / 'preview.png'), signature=key)
                summaries.append(row)
                print(f"{status} {sid} page {page_index+1}: {row['kept']}/{row['candidates']} bubbles; "
                      f"{row['covered_texts']} texts covered; {row['fallback_texts']} fallback", flush=True)
                save_json(output_root / 'index.json', summaries)
    finally:
        del detector, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return by_sequence, summaries


def replace_visuals(sequences, settings, baseline_vectors, output_root):
    """Retain every original text ID. Same bubble -> same new spatial embedding."""
    pages, summary = extract_bubbles(sequences, settings, output_root, embed=True)
    vectors, audit = {}, []
    for sequence in sequences:
        sid = sequence['sequence_id']
        vectors[sid] = {}
        for box in sequence['boxes']:
            payload = pages[sid][box['page_index']]
            bid = payload['text_to_bubble'].get(box['box_id'])
            original = baseline_vectors[sid][box['box_id']]
            vector = payload['embeddings'][bid] if bid is not None else original
            if vector.shape != original.shape:
                raise ValueError('Bubble/full-page feature dimension mismatch')
            vectors[sid][box['box_id']] = vector.clone()
            audit.append(dict(sequence_id=sid, box_id=box['box_id'], bubble_id=bid,
                feature_source='bubble_detr_encoder_mean' if bid else 'original_magi_text_query'))
    save_json(Path(output_root) / 'text_feature_audit.json', audit)
    return vectors, summary


def original_page_features(sequence, model_path, device):
    """Strict one-to-one decoder query alignment for uncovered inference text boxes."""
    import torch
    from scipy.optimize import linear_sum_assignment
    from transformers.image_transforms import center_to_corners_format
    model = load_magi(model_path, device)
    vectors, tails = {}, {}
    try:
        for page_index, page in enumerate(sequence['pages']):
            texts = [b for b in sequence['boxes'] if b['page_index'] == page_index]
            if not texts:
                continue
            with Image.open(page['path']) as image:
                array = np.asarray(image.convert('L').convert('RGB'))
            with torch.inference_mode():
                inputs = model.move_to_device(model.processor.preprocess_inputs_for_detection([array]))
                output = model._get_detection_transformer_output(**inputs)
                logits, normalized = model._get_predicted_bboxes_and_classes(output)
                tokens = model._get_predicted_obj_tokens(output)
                detected = center_to_corners_format(normalized)[0].cpu().numpy()
                detected *= np.array([page['width'], page['height'], page['width'], page['height']])
                reference = np.array([b['bbox'] for b in texts])
                intersection = np.maximum(0, np.minimum(reference[:, None, 2:], detected[None, :, 2:])-
                    np.maximum(reference[:, None, :2], detected[None, :, :2])).prod(-1)
                union = ((reference[:, 2:]-reference[:, :2]).prod(-1)[:, None] +
                         (detected[:, 2:]-detected[:, :2]).prod(-1)[None, :] - intersection)
                ious = intersection / np.maximum(union, 1e-12)
                rows, cols = linear_sum_assignment(-np.where(ious >= .7, len(texts)+1+ious, -1))
                if len(rows) != len(texts) or not (ious[rows, cols] >= .7).all():
                    raise ValueError('Original Magi text boxes cannot be aligned to local DETR queries.')
                scores, classes = logits.max(-1)
                tail_ids = model.processor._get_indices_of_tails_to_keep(scores.sigmoid(), classes.long(),
                    torch.as_tensor(detected, device=tokens.device)[None], .34)[0]
                affinity = model._get_text_tail_affinity_matrices(
                    text_obj_tokens_for_this_batch=[tokens[0, cols.tolist()]],
                    tail_obj_tokens_for_batch=[tokens[0, tail_ids]], apply_sigmoid=True)[0]
                maximum = affinity.max(1).values if len(tail_ids) else tokens.new_zeros(len(texts))
                for row, col in zip(rows, cols):
                    bid = texts[row]['box_id']
                    vectors[bid] = tokens[0, col].float().cpu()
                    score = float(maximum[row])
                    tails[bid] = torch.tensor([score, float(score > .3)])
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vectors, tails


def semantic_features(sequence, model_path, max_tokens=256, batch_size=16, device='cuda'):
    """Exactly the baseline Magi-OCR previous-two same-page target-mean pooling."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = 'right'
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device).eval()
    vectors, history, items = {}, {}, []
    try:
        for box in sequence['boxes']:
            previous = history.setdefault(box['page_index'], [])
            context = []
            for text in previous[-2:]:
                context += tokenizer.encode(text, add_special_tokens=False) + [tokenizer.sep_token_id]
            target = tokenizer.encode(box['ocr'], add_special_tokens=False) or [tokenizer.unk_token_id]
            target = target[-(max_tokens-2):]
            remaining = max_tokens-2-len(target)
            context = context[-remaining:] if remaining else []
            ids = [tokenizer.cls_token_id]+context+target+[tokenizer.sep_token_id]
            mask = [0]*(1+len(context))+[1]*len(target)+[0]
            items.append((box['box_id'], ids, mask))
            previous.append(box['ocr'])
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start+batch_size]
                length = max(len(item[1]) for item in chunk)
                ids = torch.full((len(chunk), length), tokenizer.pad_token_id, device=device, dtype=torch.long)
                attention = torch.zeros_like(ids)
                mask = torch.zeros_like(ids, dtype=torch.float)
                for i, (_, values, target_mask) in enumerate(chunk):
                    ids[i, :len(values)] = torch.tensor(values, device=device)
                    attention[i, :len(values)] = 1
                    mask[i, :len(values)] = torch.tensor(target_mask, device=device)
                hidden = model(input_ids=ids, attention_mask=attention).last_hidden_state
                pooled = ((hidden*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)).float().cpu()
                for item, vector in zip(chunk, pooled):
                    vectors[item[0]] = vector
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return vectors
