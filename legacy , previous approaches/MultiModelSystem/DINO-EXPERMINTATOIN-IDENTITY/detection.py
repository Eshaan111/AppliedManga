"""hysts YOLOv3 face detection and face crops only (no landmark model)."""
import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from identity import ROOT, save_json, export_crops

DETECTOR_SOURCE = 'hysts/anime-face-detector'
BACKENDS = ('yolov3',)
DETECTOR_PATH = ROOT / 'models/anime-face-detector-yolov3/model.safetensors'
# Optional workspace-local installation, leaving the shared CUDA environment alone.
import sys
sys.path.insert(0, str(ROOT / 'vendor'))


def expand_box(box, image_shape, scale=1.8):
    """Symmetric expansion, clipped to pixels; shape is (height, width)."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError('Expansion scale must be finite and >= 1')
    h, w = image_shape[:2]
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Expected four finite box coordinates')
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1 or h < 1 or w < 1:
        raise ValueError('Box and image must have positive dimensions')
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    result = [max(0, math.floor(cx - half_w)), max(0, math.floor(cy - half_h)),
              min(w, math.ceil(cx + half_w)), min(h, math.ceil(cy + half_h))]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('Box is outside the image')
    return result


def normalize_predictions(predictions, image_shape, confidence=.5):
    """Filter invalid/low-confidence predictions and stabilize detection IDs."""
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Confidence must be in [0, 1]')
    result = []
    for pred in predictions:
        score = float(pred['score'])
        if not math.isfinite(score) or score < confidence:
            continue
        try:
            box = expand_box(pred['box'], image_shape, 1.)
        except ValueError:
            continue
        result.append(dict(box=box, score=score, label=pred.get('label', 'face')))
    return sorted(result, key=lambda p: (*p['box'], -p['score']))


def load_detector(backend='yolov3', device=None, checkpoint_path=DETECTOR_PATH):
    import torch
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if backend == 'yolov3':
        try:
            from anime_face_detector._face import load_face_detector
        except ImportError as error:
            raise RuntimeError('Install requirements-detection.txt in a Python 3.12+ environment, '
                               'with anime-face-detector==0.1.0.') from error
        # create_detector also loads HRNet landmarks: bypass it to run one detector only.
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError('Run download_detector.py first: ' + str(checkpoint_path))
        detector = load_face_detector(backend, str(checkpoint_path), device=device)

        def predict(image):
            # hysts expects BGR arrays and returns [x1,y1,x2,y2,score].
            return [dict(box=box[:4], score=box[4], label='face')
                    for box in detector.detect(np.asarray(image)[:, :, ::-1].copy())]
    else:
        raise ValueError(f'Unknown detector: {backend}')
    return predict, device


def select_pages(annotation_path=ROOT / 'annotations/characters.json', limit=10):
    """Use existing page selection only; manual boxes/identities are not read."""
    if limit < 1:
        raise ValueError('Page limit must be positive')
    data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    task_root = Path(data['task_data_root'])
    if not task_root.is_absolute():
        task_root = ROOT / task_root
    task_root = task_root.resolve()
    pages, seen = [], set()
    for seq in data['sequences']:
        sid = seq['sequence_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
            raise ValueError('Sequence IDs must be filesystem-safe')
        for page in seq['pages']:
            key = sid, page['page_index']
            if key in seen or not isinstance(key[1], int) or key[1] < 0:
                raise ValueError(f'Duplicate or invalid page: {key}')
            seen.add(key)
            path = (task_root / page['image']).resolve()
            if not path.is_relative_to(task_root):
                raise ValueError('Page path escapes TaskData')
            if not path.is_file():
                raise FileNotFoundError(path)
            pages.append(dict(sequence_id=sid, split=seq.get('split', 'calibration'),
                              page_index=page['page_index'], image_path=str(path)))
    pages.sort(key=lambda p: (p['sequence_id'], p['page_index']))
    if not pages:
        raise ValueError('No pages in page selection')
    return pages[:limit]


def draw_boxes(image, detections, output):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for pred in detections:
        draw.rectangle(pred['box'], outline='red', width=3)
        draw.text((pred['box'][0], max(0, pred['box'][1] - 14)),
                  f"{pred['crop_id']} {pred['score']:.2f}", fill='red')
    canvas.save(output)


def run_detection(pages, output, backend='yolov3', confidence=.5,
                  device=None, predictor=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    # Validate settings even when the detector finds no faces.
    normalize_predictions([], (1, 1), confidence)
    if not pages:
        raise ValueError('No input pages')
    if predictor is None:
        predictor, device = load_detector(backend, device)
    output.mkdir(parents=True)
    (output / 'overlays').mkdir()
    rows, page_results, review = [], [], []
    for page in pages:
        sid, pi = page['sequence_id'], page['page_index']
        with Image.open(page['image_path']) as source:
            image = source.convert('RGB')
        start = time.perf_counter()
        predictions = normalize_predictions(predictor(image), image.size[::-1], confidence)
        elapsed = time.perf_counter() - start
        detections = []
        for n, pred in enumerate(predictions):
            cid = f'p{pi}_c{n:03d}'
            detections.append(dict(pred, crop_id=cid))
            rows.append(dict(page, crop_id=cid, strategy='face', identity_label=None,
                             box=pred['box'], face_box=pred['box'], detector_score=pred['score']))
        overlay = output / 'overlays' / f'{sid}_p{pi}.png'
        draw_boxes(image, detections, overlay)
        page_results.append(dict(page, detections=detections, seconds=elapsed, overlay_path=str(overlay)))
        review.append(dict(sequence_id=sid, page_index=pi, detected_faces=len(detections),
                           missed_faces=None, false_positives=None, notes=''))
        print(f'{backend}: {sid} page {pi}: {len(detections)} faces ({elapsed:.2f}s)', flush=True)
    rows = export_crops(rows, output / 'crops')
    manifest = dict(schema_version=1, backend=backend, pages=page_results, rows=rows,
                    config=dict(confidence=confidence, device=device,
                                model_id=DETECTOR_SOURCE, architecture=backend, landmarks=False,
                                checkpoint_path=str(DETECTOR_PATH)))
    save_json(output / 'detections.json', manifest)
    save_json(output / 'detection_review.json', review)
    save_json(output / 'identity_labels.json', {sid: {r['crop_id']: '' for r in rows if r['sequence_id'] == sid}
                                               for sid in {p['sequence_id'] for p in pages}})
    save_json(output / 'summary.json', dict(pages=len(pages), faces=len(rows),
              zero_detection_pages=sum(not p['detections'] for p in page_results),
              inference_seconds=sum(p['seconds'] for p in page_results),
              quality='Unmeasured until overlays are manually checked'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT / 'annotations/characters.json')
    parser.add_argument('--detector', choices=BACKENDS, default='yolov3')
    parser.add_argument('--pages', type=int, default=10)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--device')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'outputs' / f'detection_{args.detector}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run_detection(select_pages(args.annotations, args.pages), output, args.detector,
                  args.confidence, args.device)
    print(f'Saved: {output}')
