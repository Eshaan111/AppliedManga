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
    'cache_dir': ROOT / 'cache_selected_ocr', 'run_dir': ROOT / 'runs_selected_ocr',
    'seed': 42, 'val_fraction': 0.2, 'beta': 0.999,
    'narration_sampling_weight': 2.0,
    'batch_size': 4, 'epochs': 40, 'lr': 0.0003,
    'max_tokens': 256, 'semantic_batch_size': 16,
    'match_threshold': 95.0, 'match_margin': 5.0,
    'min_bbox_iou': 0.7, 'text_threshold': 0.3,
    'tail_detection_threshold': 0.34, 'tail_match_threshold': 0.3,
    'grayscale': True,
}


BASELINE_ROOT = ROOT.parent / 'NarrationImbalanceTraining'
BUBBLE_SETTINGS = dict(
    bubble_model=r'C:/MODELS/manga_bubbles/bubbles_detect.pt',
    bubble_imgsz=1024, bubble_confidence=0.25, text_coverage=0.8,
    magi_model=CONFIG['magi_model'], device='cuda',
    pooling='detr_encoder_spatial_mean_v1', preprocessing='grayscale_RGB',
    fallback='original_magi_text_query', ocr_source='selected_magi_paddle')
CONFIG['bubble_settings'] = BUBBLE_SETTINGS

PADDLE_SETTINGS = dict(lang='en', device='gpu:0', ocr_version='PP-OCRv5',
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
PADDLE_DET_PATH = Path.home() / '.paddlex/official_models/PP-OCRv5_server_det'
PADDLE_REC_PATH = Path.home() / '.paddlex/official_models/en_PP-OCRv5_mobile_rec'
PADDLE_LOCAL_SETTINGS = dict(text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',
    text_detection_model_dir=str(PADDLE_DET_PATH), text_recognition_model_dir=str(PADDLE_REC_PATH))
PREFERRED_OCR = 'paddle'

# Match the selected standard classifier architecture for the first controlled comparison.
CONFIG.update(lr=0.001338508350120473, dropout=0.3, num_heads=8,
              dim_feedforward=256, weight_decay=0.013988081457939344,
              narration_sampling_weight=2.5)

# Shared output paths used by notebooks 05, 06 and 07.
TRAINING_PATHS = {
    'deberta_run': ROOT / 'runs_deberta_resume_v1',
    'classifier_run': ROOT / 'runs_classifier_resume_v1_tuned_Bert',
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
