"""Run bubble/DETR extraction, never classifier training. Resume with the same command."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from pathlib import Path
import argparse
import contextlib
import io
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import CONFIG, BUBBLE_SETTINGS, BASELINE_ROOT, run_stage
from bubble_features import extract_bubbles, sha256


def test_sequence(sequence_id, split='test'):
    from PIL import Image
    magi_root = ROOT.parent.parent
    folder = magi_root / 'TaskData' / split / 'images' / sequence_id
    cache = magi_root / 'paddleOCR-Magi/cache_inference' / split / sequence_id
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    raw = json.loads((cache / 'magi_raw.json').read_text(encoding='utf-8'))
    # Existing cache was produced from original Magi OCR, not the selected OCR stage.
    from config import fingerprint
    if raw['data_hash'] != fingerprint(raw['data']):
        raise ValueError('Corrupt raw Magi cache')
    inputs = manifest['inputs']
    if inputs['sequence_id'] != sequence_id or inputs['split'] != split:
        raise ValueError('Wrong sequence cache')
    paths = sorted(folder.glob('*.png'))
    if len(paths) != len(raw['data']) or len(paths) != len(inputs['pages']):
        raise ValueError('Raw cache page count mismatch')
    pages, boxes = [], []
    for i, (path, result, meta) in enumerate(zip(paths, raw['data'], inputs['pages'])):
        if sha256(path) != meta['sha256'] or path.name != meta['name']:
            raise ValueError('Source page no longer matches the raw cache')
        with Image.open(path) as image:
            width, height = image.size
        if len(result['texts']) != len(result['ocr']):
            raise ValueError('Magi text/OCR mismatch')
        pages.append(dict(path=str(path), sha256=sha256(path), width=width, height=height))
        for j, (box, ocr) in enumerate(zip(result['texts'], result['ocr'])):
            boxes.append(dict(box_id=f'PG{i+1}:T{j+1}', page_index=i, text_index=j,
                              bbox=box, ocr=ocr, label=-100, gt_speaker=None))
    return dict(sequence_id=sequence_id, pages=pages, boxes=boxes, raw_cache=str(cache/'magi_raw.json'))


def run_original_notebook():
    """Run the standalone detector copy with explicit local model and page paths."""
    source = ROOT / 'BubbleDetector_executed_smoke.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), str(source), 'exec'), namespace)
    print('Detected bubbles:', len(namespace['bubbles']))
    namespace.clear()
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['test', 'development', 'both'], default='both')
    parser.add_argument('--sequence-id', default='seq_14469aeb38963085')
    args = parser.parse_args()
    if args.split in {'test', 'both'}:
        run_original_notebook()
        sequence = test_sequence(args.sequence_id)
        output = CONFIG['cache_dir'] / 'bubbles/test'
        output.mkdir(parents=True, exist_ok=True)
        from ocr_selection import select_sequences
        selected, _ = select_sequences([sequence])
        (output / 'sequence.json').write_text(json.dumps(selected[0], indent=2), encoding='utf-8')
        extract_bubbles(selected, BUBBLE_SETTINGS, output)
    if args.split in {'development', 'both'}:
        cache = CONFIG['cache_dir']
        cache.mkdir(parents=True, exist_ok=True)
        run_stage('01_prepare_sequence_cache.ipynb')
        run_stage('04_build_training_samples.ipynb')


if __name__ == '__main__':
    main()
