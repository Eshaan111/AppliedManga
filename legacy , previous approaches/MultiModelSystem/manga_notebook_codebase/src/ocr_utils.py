"""Label-independent OCR extraction with fingerprinted, resumable output."""
import hashlib
import json
import math
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def discover_sequences(source, split='all'):
    """Use manifest image order, or lexically sorted three-page folders."""
    source = Path(source).resolve()
    manifest = source / 'sequences.json'
    if manifest.is_file():
        entries = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(entries, list):
            raise ValueError('sequences.json must be a list')
        entries = [e for e in entries if split == 'all' or e.get('split') == split]
    else:
        if split != 'all':
            raise ValueError('--split requires sequences.json')
        image_root = source / 'images' if (source / 'images').is_dir() else source
        entries = [dict(sequence_id=folder.name, images=[str(p.relative_to(source)) for p in
                   sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
                   for folder in sorted(image_root.iterdir()) if folder.is_dir()
                   and any(p.suffix.lower() in IMAGE_EXTS for p in folder.iterdir())]
    if not entries:
        raise ValueError('No sequences found')
    seen, rows = set(), []
    for entry in entries:
        sid = entry['sequence_id']
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError(f'Invalid or duplicate sequence ID: {sid!r}')
        seen.add(sid)
        if len(entry['images']) != 3:
            raise ValueError(f'{sid}: expected exactly three images')
        paths = [(source / relative).resolve() for relative in entry['images']]
        if len(set(paths)) != 3:
            raise ValueError(f'{sid}: repeated image path')
        for path in paths:
            if not path.is_relative_to(source) or not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f'Missing or unsafe image: {path}')
        rows.append(dict(sequence_id=sid, images=paths))
    return rows


def ocr_page(ocr, path):
    rows = []
    for result in ocr.predict(str(path)):
        data = result.json
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        data = data.get('res', data)
        texts, scores = data['rec_texts'], data['rec_scores']
        boxes = data.get('rec_boxes')
        if boxes is None:
            boxes = [[min(p[0] for p in poly), min(p[1] for p in poly),
                      max(p[0] for p in poly), max(p[1] for p in poly)] for poly in data['rec_polys']]
        if not len(texts) == len(scores) == len(boxes):
            raise ValueError('OCR returned inconsistent arrays')
        for text, score, box in zip(texts, scores, boxes):
            box, score = [float(v) for v in box], float(score)
            if len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]):
                raise ValueError('OCR returned invalid coordinates or score')
            rows.append(dict(id=f'c{len(rows)}', text=str(text), score=score, box=box))
    return rows


def extract_sequences(rows, ocr, output, runtime):
    """Resume only if image, runtime, code, and cached-output hashes match."""
    output = Path(output)
    metadata_path = output.with_suffix('.metadata.json')
    cached, metadata = {}, dict(runtime=runtime, sequences={})
    if output.exists():
        if not metadata_path.exists():
            raise ValueError('Legacy cache has no provenance; preserve it and choose a new --output')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['runtime'] != runtime or metadata.get('output_sha256') != sha256(output):
            raise ValueError('Runtime or output changed; choose a new --output')
        for line in output.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            sid = record['sequence_id']
            if sid in cached or not isinstance(record['pages'], list) or len(record['pages']) != 3:
                raise ValueError('Invalid cached sequence')
            cached[sid] = record
    fingerprints = {row['sequence_id']: [sha256(p) for p in row['images']] for row in rows}
    for sid, hashes in fingerprints.items():
        if sid in cached and metadata['sequences'].get(sid) != hashes:
            raise ValueError(f'{sid}: images changed; choose a new --output')
    for row in rows:
        sid = row['sequence_id']
        if sid in cached:
            print(f'Reused {sid}', flush=True)
            continue
        pages = [ocr_page(ocr, path) for path in row['images']]
        cached[sid] = dict(sequence_id=sid, pages=pages)
        metadata['sequences'][sid] = fingerprints[sid]
        atomic_write(output, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cached.values()))
        metadata['output_sha256'] = sha256(output)
        atomic_write(metadata_path, json.dumps(metadata, indent=2) + '\n')
        print(f'Extracted {sid}: {sum(map(len, pages))} candidates', flush=True)
    return cached


def ensure_ocr_cache(root, rows, split='val', cache_path=None):
    """Notebook bridge; existing legacy OCR remains readable without modification.

    New extraction uses the standalone CLI and selected image IDs, never targets.
    Set cache_path to a fresh output to regenerate legacy entries with provenance.
    """
    import os
    import subprocess
    root = Path(root).resolve()
    path = Path(cache_path) if cache_path is not None else root / 'outputs/ocr/paddleocr/candidates.jsonl'
    cached = {}
    if path.exists():
        cached = {r['sequence_id']: r for r in
                  (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}
        metadata_path = path.with_suffix('.metadata.json')
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata.get('output_sha256') != sha256(path):
                raise ValueError('Cached output changed; choose a new cache_path')
            for row in rows:
                sid = row['sequence_id']
                if sid in cached:
                    hashes = [sha256(root / 'TaskData' / p) for p in row['images']]
                    if hashes != metadata['sequences'].get(sid):
                        raise ValueError(f'{sid}: images changed; choose a new cache_path')
    missing = [row['sequence_id'] for row in rows if row['sequence_id'] not in cached]
    if missing:
        if path.exists() and not path.with_suffix('.metadata.json').exists():
            raise ValueError('Legacy cache is incomplete; set cache_path to a new output for reproducible extraction')
        interpreter = Path(os.environ.get('MANGA_OCR_PYTHON', str(root.parent / '.venvOCR/Scripts/python.exe')))
        if not interpreter.is_file():
            raise RuntimeError('Set MANGA_OCR_PYTHON to the Python 3.11 OCR interpreter')
        command = [str(interpreter), str(root / 'scripts/build_ocr_cache.py'),
                   '--input', str(root / 'TaskData'), '--output', str(path)]
        for sid in missing:
            command.extend(['--sequence-id', sid])
        subprocess.run(command, check=True)
        cached = {r['sequence_id']: r for r in
                  (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}
    return {row['sequence_id']: cached[row['sequence_id']]['pages'] for row in rows}
