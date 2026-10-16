from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MAGI_ROOT = Path(r'C:\Users\ESHAAN\BYTE\Applied-Task\Magi')
CONFIG = {
    'labels_jsonl': MAGI_ROOT / 'TaskData/development/labels.jsonl',
    'images_root': MAGI_ROOT / 'TaskData/development/images',
    'raw_cache_roots': [MAGI_ROOT / 'paddleOCR-Magi/cache_baseline/development',
                        MAGI_ROOT / 'FINETUNING-EXPERMENTS/TextRoleVL/cache/development'],
    'legacy_visual_cache': MAGI_ROOT / 'FINETUNING-EXPERMENTS/isNarratorTextMagi/narration_data.pt',
    'magi_model': r'C:\MODELS\magiv2',
    'deberta_model': r'C:\MODELS\deberta-v3-base',
    'dino_model': r'C:\MODELS\dinov2-small',
    'crop_padding': 0.20,  # Fraction of original width/height added on EACH side.
    'crop_image_size': 224,  # Divisible by DINOv2's patch size 14.
    'crop_batch_size': 32,
    'base_experiment': ROOT.parent / 'NarrationImbalanceTraining',
    'cache_dir': ROOT / 'cache', 'run_dir': ROOT / 'runs',
    'seed': 42, 'val_fraction': 0.2, 'beta': 0.999,
    'narration_sampling_weight': 2.0,
    'batch_size': 4, 'epochs': 20, 'lr': 0.0003,
    'max_tokens': 256, 'semantic_batch_size': 16,
    'match_threshold': 95.0, 'match_margin': 5.0,
    'min_bbox_iou': 0.7, 'text_threshold': 0.3,
    'tail_detection_threshold': 0.34, 'tail_match_threshold': 0.3,
    'grayscale': True,
}


def fingerprint(value):
    text = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def model_files(path):
    path = Path(path)
    if not path.is_dir():
        return str(path)
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(path.iterdir())
            if p.suffix in {'.py', '.json', '.bin', '.safetensors'}}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run_stage(filename):
    # Only used to create or refresh prerequisite caches; the notebook prints what runs.
    print('Running prerequisite:', filename)
    doc = json.loads((ROOT / filename).read_text(encoding='utf-8'))
    namespace = {'CONFIG': CONFIG, '__name__': '__main__'}
    for i, cell in enumerate(doc['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'{filename}:cell{i}', 'exec'), namespace)
    return namespace


def require_cache(path, stage):
    if not path.exists():
        print('Missing cache:', path)
        run_stage(stage)
    assert path.exists(), f'Run {stage} to create {path}'


def reuse_base_caches():
    """Copy only missing compatible starting caches; never copy old training results."""
    import shutil
    CONFIG['cache_dir'].mkdir(parents=True, exist_ok=True)
    mapping = {'sequences.json': 'sequences.json', 'deberta.pt': 'deberta.pt',
               'magi.pt': 'magi.pt', 'samples.pt': 'base_samples.pt'}
    for old_name, new_name in mapping.items():
        source = Path(CONFIG['base_experiment']) / 'cache' / old_name
        destination = CONFIG['cache_dir'] / new_name
        if not destination.exists() and source.exists():
            shutil.copy2(source, destination)
            print('Copied existing base cache:', new_name)
