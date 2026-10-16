"""Headless single-sequence runtime extracted from the selected notebook.

Called by run_sequence_inference.py; no notebook is read or executed at runtime.
The contour helpers remain in paddleOCR-Magi. Generation/classification rules are preserved.
"""
import json
import sys
from pathlib import Path

if __name__ != "__main__":
    raise RuntimeError("Use run_sequence_inference.py, rather than importing this worker.")
BATCH_SETTINGS = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

# Notebook source SHA256 at extraction: 9e83c9f4117d2413b0abeef0019b24078f38fa2da35628efd1de016adb0c3360


# --- Notebook inference cell 2 ---
# Set offline mode before importing Hugging Face model loaders.
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
import torch  # Must precede Paddle on this Windows installation.
import paddle
import gc
import copy
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoConfig
from paddleocr import PaddleOCR

print('Python:', sys.executable)
print('Torch:', torch.__version__, '| Paddle:', paddle.__version__)

import importlib.util
import inspect
import types
from scipy.optimize import linear_sum_assignment

# Load a local Python file under a unique module name: avoid collisions with other config/model files.
def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import math
import re
from difflib import SequenceMatcher
from io import BytesIO
import cv2

# Batch mode exports audits instead of notebook tables or widgets.
def display(*args, **kwargs):
    pass

# The stage scheduler supplies this shared dictionary; standalone runs keep old loading behavior.
SHARED_MODELS = globals().get('SHARED_MODELS')
ALLOWED_STAGES = globals().get('ALLOWED_STAGES')


def cache_message(status, name):
    message = f'{status} {name}'
    print(message, flush=True)
    # Keep full logs, but show only this phase's cache decisions in the terminal.
    stream = globals().get('CACHE_STATUS_STREAM')
    if stream is not None and (ALLOWED_STAGES is None or name in ALLOWED_STAGES):
        print(f'    {message}', file=stream, flush=True)


def load_model_once(name, factory):
    if SHARED_MODELS is None:
        return factory()
    if name not in SHARED_MODELS:
        SHARED_MODELS[name] = factory()
    return SHARED_MODELS[name]


# --- Notebook inference cell 4 ---
# Local resource paths: these are the only model/checkpoint resources required.
MAGI_ROOT=Path(BATCH_SETTINGS['magi_root'])
PIPELINE_ROOT=MAGI_ROOT/'paddleOCR-Magi'
# Model source: https://huggingface.co/ragavsachdeva/magiv2
MODEL_PATH=Path('C:/MODELS/magiv2')
# Model source: https://huggingface.co/google/gemma-3-4b-it
GEMMA_MODEL_PATH=Path('C:/MODELS/gemma-3-4b-it')
# Bubble checkpoint source: https://huggingface.co/PSImera/manga_bubbles_detect
BUBBLE_MODEL=Path('C:/MODELS/manga_bubbles/bubbles_detect.pt')
# Our fine-tuned checkpoint is not published yet; base: https://huggingface.co/ragavsachdeva/magiv2
CHECKPOINT_PATH=MAGI_ROOT/'FINETUNING-EXPERMENTS/fineTuningMagiCharSimHeads2/magi_crop_last_2_blocks.pt'
# Model source: https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
PADDLE_DET_PATH=Path.home()/'.paddlex/official_models/PP-OCRv5_server_det'
# Model source: https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec
PADDLE_REC_PATH=Path.home()/'.paddlex/official_models/en_PP-OCRv5_mobile_rec'

# Use TaskData, or set INPUT_IMAGES to an ordered list of one to three image paths.
SPLIT='development'
SEQUENCE_ID='seq_2fae297a18af2a57'
PAGE_NUMBER=None  # Dataset mode only: None selects the complete sequence.
INPUT_IMAGES=None
# INPUT_IMAGES=[Path('C:/path/page1.png'),Path('C:/path/page2.png')]
CACHE_ROOT=PIPELINE_ROOT/'cache_artifact_contour_full_inference'
FORCE_ALL=False
CLEAN_TRANSCRIPT=True
SAVE_ANNOTATED_PAGES=False

# Batch inputs/resources; prediction settings below retain notebook defaults.
SPLIT=BATCH_SETTINGS['split']
SEQUENCE_ID=BATCH_SETTINGS['sequence_id']
INPUT_IMAGES=[Path(path) for path in BATCH_SETTINGS['images']]
PAGE_NUMBER=None
CACHE_ROOT=Path(BATCH_SETTINGS['cache_root'])
FORCE_ALL=BATCH_SETTINGS.get('force',False)
CLEAN_TRANSCRIPT=BATCH_SETTINGS.get('clean_transcript',True)
SAVE_ANNOTATED_PAGES=False
MODEL_PATH=Path(BATCH_SETTINGS.get('magi_model',MODEL_PATH))
GEMMA_MODEL_PATH=Path(BATCH_SETTINGS.get('gemma_model',GEMMA_MODEL_PATH))
BUBBLE_MODEL=Path(BATCH_SETTINGS.get('bubble_model',BUBBLE_MODEL))
CHECKPOINT_PATH=Path(BATCH_SETTINGS.get('character_checkpoint',CHECKPOINT_PATH))
PADDLE_DET_PATH=Path(BATCH_SETTINGS.get('paddle_detector',PADDLE_DET_PATH))
PADDLE_REC_PATH=Path(BATCH_SETTINGS.get('paddle_recognizer',PADDLE_REC_PATH))

# Existing character pipeline settings/checkpoint stay unchanged.
ENTITY_THRESHOLD=0.82
THRESHOLD_CHECKPOINT_EPOCH=20
CHARACTER_CLEANUP_SETTINGS=dict(min_iou=.90,min_cosine=.95,algorithm='contained_direct_outer_match_v1')
ASSOCIATION_HEAD_PATH=None  # Original Magi association head by default.
MODEL_REVISION='local-magiv2-1'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LANGUAGE='en'
PADDLE_SETTINGS=dict(lang=LANGUAGE,device='gpu:0',ocr_version='PP-OCRv5',
    use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
PADDLE_LOCAL_SETTINGS=dict(text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',
    text_detection_model_dir=str(PADDLE_DET_PATH),text_recognition_model_dir=str(PADDLE_REC_PATH))
DETECTION=dict(imgsz=1024,confidence=.25)
CONTOUR_MODULE_PATH=PIPELINE_ROOT/'contour_narrator2.py'
CONTOUR_GROUPING_PATH=PIPELINE_ROOT/'contour_grouping.py'
contour_model=load_local_module('standalone_artifact_contour_rules',CONTOUR_MODULE_PATH)
CONTOUR_SETTINGS=dict(contour_model.CHECK)
REGION_SETTINGS=dict(contour_model.REGION)
FAINT_REGION_SETTINGS=dict(contour_model.FAINT_REGION)
GROUP_FAINT_SETTINGS=dict(contour_model.contours.FAINT)
contour_model.CHECK=CONTOUR_SETTINGS;contour_model.REGION=REGION_SETTINGS
contour_model.FAINT_REGION=FAINT_REGION_SETTINGS;contour_model.contours.FAINT=GROUP_FAINT_SETTINGS
STAGES=['magi_raw','embeddings','paddle_ocr','ocr_selection','artifact_labels',
    'artifact_filter','character_cleanup','entities','text_entities','contour_bubbles',
    'contour_roles','contour_speakers','gemma_transcript']
FORCE={name:FORCE_ALL and (ALLOWED_STAGES is None or name in ALLOWED_STAGES) for name in STAGES}
STAGE_VERSIONS={name:1 for name in STAGES}


# --- Notebook inference cell 6 ---
SPLIT=str(SPLIT).strip();SEQUENCE_ID=str(SEQUENCE_ID).strip()
assert SEQUENCE_ID and Path(SEQUENCE_ID).name==SEQUENCE_ID,'Use a single sequence name, not a path'
if INPUT_IMAGES is None:
    SEQUENCE_ROOT=MAGI_ROOT/'TaskData'/SPLIT/'images'/SEQUENCE_ID
    assert SEQUENCE_ROOT.is_dir(),f'Missing image directory: {SEQUENCE_ROOT}'
    sequence_files=sorted(p for p in SEQUENCE_ROOT.iterdir() if p.suffix.lower()=='.png' and p.stem.isdigit())
    PAGE_FILES=sequence_files if PAGE_NUMBER is None else [p for p in sequence_files if int(p.stem)==int(PAGE_NUMBER)]
    PAGE_LABELS=[int(p.stem) for p in PAGE_FILES]
else:
    PAGE_FILES=[Path(p).expanduser().resolve() for p in INPUT_IMAGES]
    sequence_files=list(PAGE_FILES)
    PAGE_LABELS=list(range(1,len(PAGE_FILES)+1))
assert 1<=len(PAGE_FILES)<=3,'Supply one to three ordered page images'
assert len({str(p.resolve()) for p in PAGE_FILES})==len(PAGE_FILES),'Repeated input image path'
assert len(PAGE_LABELS)==len(set(PAGE_LABELS)),'Repeated page ID'
for path in PAGE_FILES:assert path.is_file(),f'Missing input image: {path}'
for path in [MODEL_PATH,GEMMA_MODEL_PATH,PADDLE_DET_PATH,PADDLE_REC_PATH]:
    assert path.is_dir(),f'Missing local model directory: {path}'
for path in [CHECKPOINT_PATH,BUBBLE_MODEL,CONTOUR_MODULE_PATH,CONTOUR_GROUPING_PATH]:
    assert path.is_file(),f'Missing local resource: {path}'
if ASSOCIATION_HEAD_PATH is not None:
    ASSOCIATION_HEAD_PATH=Path(ASSOCIATION_HEAD_PATH);assert ASSOCIATION_HEAD_PATH.is_file()
chapter_pages=[];crop_pages=[]
for path in PAGE_FILES:
    with Image.open(path) as image:
        chapter_pages.append(np.asarray(image.convert('L').convert('RGB')))
        crop_pages.append(np.asarray(image.convert('RGB')))
# A deterministic scope separates different selected image lists; all reuse stays inside CACHE_ROOT.
scope=hashlib.sha256(json.dumps([str(p.resolve()) for p in PAGE_FILES]).encode()).hexdigest()[:16]
CACHE_DIR=CACHE_ROOT/SEQUENCE_ID/scope
OUTPUT_DIR=Path(BATCH_SETTINGS['sequence_output_dir'])
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
print('Images:',[str(p) for p in PAGE_FILES])
print('Own cache only:',CACHE_DIR)


# --- Notebook inference cell 8 ---
def json_ready(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def digest(data):
    text = json.dumps(json_ready(data), sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_ready(data), ensure_ascii=True,
                                    indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def model_fingerprint():
    files = []
    for path in sorted(MODEL_PATH.rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.py', '.safetensors', '.bin', '.pt', '.pth'}:
            stat = path.stat()
            files.append(dict(name=str(path.relative_to(MODEL_PATH)), size=stat.st_size,
                              mtime_ns=stat.st_mtime_ns,
                              sha256=file_hash(path) if path.suffix in {'.json', '.py'} else None))
    return dict(path=str(MODEL_PATH), revision=MODEL_REVISION, files=files)


PACKAGES = {name: version(name) for name in ['torch', 'torchvision', 'transformers',
            'paddlepaddle-gpu', 'paddleocr', 'paddlex', 'numpy', 'pillow']}
PAGE_METADATA = [dict(name=p.name, sha256=file_hash(p), width=image.shape[1], height=image.shape[0])
                 for p, image in zip(PAGE_FILES, chapter_pages)]
INPUTS = dict(sequence_id=SEQUENCE_ID, split=SPLIT, pages=PAGE_METADATA,
              preprocessing='grayscale_then_RGB')
MODEL_INFO = model_fingerprint()
if ASSOCIATION_HEAD_PATH is not None:
    MODEL_INFO['association_head_sha256'] = file_hash(ASSOCIATION_HEAD_PATH)
# Our trained two-block checkpoint; original base: https://huggingface.co/ragavsachdeva/magiv2
checkpoint = load_model_once('character_checkpoint', lambda: torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True))
assert checkpoint['version'] == 2 and checkpoint['settings']['trained_blocks'] == 2
assert checkpoint['best_epoch'] == THRESHOLD_CHECKPOINT_EPOCH, (
    'Checkpoint epoch changed. Set ENTITY_THRESHOLD and THRESHOLD_CHECKPOINT_EPOCH from its validation clustering run.')
for name, expected in checkpoint['base_model_hashes'].items():
    assert file_hash(MODEL_PATH / name) == expected, f'Original model changed: {name}'
CHECKPOINT_INFO = dict(path=str(CHECKPOINT_PATH), sha256=file_hash(CHECKPOINT_PATH),
                       epoch=checkpoint['best_epoch'], trained_blocks=2)
manifest = dict(schema_version=1, inputs=INPUTS, model=MODEL_INFO, packages=PACKAGES,
                crop_encoder=CHECKPOINT_INFO, stages={})
print('Crop checkpoint epoch:', checkpoint['best_epoch'], '| clustering threshold:', ENTITY_THRESHOLD)
MANIFEST_PATH = CACHE_DIR / 'manifest.json'
if MANIFEST_PATH.exists():
    try:
        old_manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        if old_manifest.get('inputs') == INPUTS:
            manifest['stages'] = old_manifest.get('stages', {})
    except (ValueError, OSError):
        pass


def stage_ref(name):
    entry = manifest['stages'][name]
    return dict(signature=entry['signature'], data_hash=entry['data_hash'])


def cached_stage(name, settings, compute):
    signature = digest(dict(inputs=INPUTS, algorithm_version=STAGE_VERSIONS[name], settings=settings))
    path = CACHE_DIR / f'{name}.json'
    payload = None
    if path.exists() and not FORCE[name]:
        try:
            candidate = json.loads(path.read_text(encoding='utf-8'))
            if candidate['signature'] == signature and digest(candidate['data']) == candidate['data_hash']:
                payload = candidate
        except (ValueError, KeyError, OSError):
            pass
    if payload is None:
        if ALLOWED_STAGES is not None and name not in ALLOWED_STAGES:
            raise RuntimeError(f'Prior stage {name} is missing/stale; restart the batch from its first phase')
        cache_message('MISS', name)
        data = json_ready(compute())
        payload = dict(signature=signature, data_hash=digest(data),
                       created_at=datetime.now(timezone.utc).isoformat(), data=data)
        write_json(path, payload)
    else:
        cache_message('HIT', name)
    manifest['stages'][name] = dict(file=path.name, signature=signature,
                                  data_hash=payload['data_hash'], settings=settings,
                                  algorithm_version=STAGE_VERSIONS[name], created_at=payload['created_at'])
    write_json(MANIFEST_PATH, manifest)
    return payload['data']

def function_digest(function):
    # Function source is added to new stages' cache keys, so notebook edits invalidate their results.
    # Ignore notebook execution filenames/line numbers; they change across kernel restarts.
    def describe(code):
        constants = [describe(value) if isinstance(value, types.CodeType) else repr(value)
                     for value in code.co_consts]
        return dict(bytecode=code.co_code.hex(), names=code.co_names,
                    variables=code.co_varnames, constants=constants)
    return digest(describe(inspect.unwrap(function).__code__))

DEPENDENCIES=[dict(path=str(path.resolve()),sha256=file_hash(path),role=role)
    for path,role in [(CONTOUR_MODULE_PATH,'background regions and SPOKEN/NARRATED'),
                      (CONTOUR_GROUPING_PATH,'individual boundaries, faint retry and overlap groups')]]
GEMMA_IDENTITY=dict(path=str(GEMMA_MODEL_PATH),files={
    str(p.relative_to(GEMMA_MODEL_PATH)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
        sha256=file_hash(p) if p.suffix=='.json' else None)
    for p in sorted(GEMMA_MODEL_PATH.rglob('*')) if p.is_file() and p.suffix in {'.json','.safetensors','.bin','.model'}})
RESOURCE_MANIFEST=dict(helpers=DEPENDENCIES,
    models=dict(magi=MODEL_INFO,character_checkpoint=CHECKPOINT_INFO,gemma=GEMMA_IDENTITY,
        bubble=dict(path=str(BUBBLE_MODEL),sha256=file_hash(BUBBLE_MODEL)),paddle=PADDLE_LOCAL_SETTINGS),
    cache_policy='Read/write CACHE_ROOT only; no external cache discovery',
    input_images=[str(p.resolve()) for p in PAGE_FILES])
write_json(OUTPUT_DIR/'dependencies.json',RESOURCE_MANIFEST)


# --- Notebook inference cell 10 ---
magi_model = None


def get_magi():
    global magi_model
    if magi_model is None:
        assert torch.cuda.is_available(), 'Torch CUDA unavailable'
        # Normal tensors allow the later checkpoint copy; only predictions use inference mode.
        with torch.inference_mode(False):
            # Base model: https://huggingface.co/ragavsachdeva/magiv2
            config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
            # The full checkpoint already contains the backbone: avoid timm downloading it again.
            config.detection_model_config.use_pretrained_backbone = False
            config.disable_ocr = False
            config.disable_crop_embeddings = False
            magi_model, info = load_model_once('magi', lambda: AutoModel.from_pretrained(str(MODEL_PATH), config=config,
                trust_remote_code=True, local_files_only=True, output_loading_info=True))
            assert not info['missing_keys'] and not info.get('mismatched_keys'), info
            magi_model = magi_model.cuda().eval()
            magi_model.requires_grad_(False)
            if ASSOCIATION_HEAD_PATH is not None:
                association = torch.load(ASSOCIATION_HEAD_PATH, map_location='cpu', weights_only=True)
                magi_model.text_character_matching_head.load_state_dict(association['state_dict'], strict=True)
    return magi_model


def predict_magi():
    model = get_magi()
    with torch.inference_mode():
        return model.do_chapter_wide_prediction(
            chapter_pages, {'images': [], 'names': []}, use_tqdm=True, do_ocr=True)


def predict_embeddings():
    boxes = [result['characters'] for result in magi_raw_results]
    if not any(boxes):
        return [[] for _ in boxes]
    model = get_magi()
    last = len(model.crop_embedding_model.encoder.layer) - 1
    assert last >= 1
    prefixes = (f'crop_embedding_model.encoder.layer.{last-1}.',
                f'crop_embedding_model.encoder.layer.{last}.')
    parameters = dict(model.named_parameters())
    expected_names = {name for name in parameters if name.startswith(prefixes)}
    assert tuple(checkpoint['trainable_prefixes']) == prefixes
    assert set(checkpoint['state_dict']) == expected_names
    # Patch only the two crop blocks after original Magi prediction is complete.
    with torch.no_grad():
        for name, tensor in checkpoint['state_dict'].items():
            assert tensor.shape == parameters[name].shape
            parameters[name].copy_(tensor.to(parameters[name].device))
    model.eval()
    with torch.inference_mode():
        embeddings = model.predict_crop_embeddings(crop_pages, boxes, mask_ratio=0.0, batch_size=32)
        return [torch.nn.functional.normalize(x, p=2, dim=1) if len(x) else x for x in embeddings]


def validate_magi(results):
    assert len(results) == len(chapter_pages), 'Magi page count mismatch'
    for result in results:
        assert 'tails' in result and 'text_tail_associations' in result,'Magi must return tail detections/links'
        assert len(result['texts']) == len(result['ocr']), 'Text/OCR count mismatch'
        assert len(result['characters']) == len(result['character_cluster_labels']), 'Character/cluster count mismatch'
        for text_index, char_index in result['text_character_associations']:
            assert 0 <= text_index < len(result['texts']), 'Invalid text association index'
            assert 0 <= char_index < len(result['characters']), 'Invalid character association index'


# --- Notebook inference cell 12 ---
try:
    magi_raw_results = cached_stage('magi_raw',
        dict(model=MODEL_INFO, torch=PACKAGES['torch'], torchvision=PACKAGES['torchvision'],
             transformers=PACKAGES['transformers'], do_ocr=True), predict_magi)
    validate_magi(magi_raw_results)
    embeddings_per_page = cached_stage('embeddings',
        dict(magi=stage_ref('magi_raw'), model=MODEL_INFO, checkpoint=CHECKPOINT_INFO,
             crop_preprocessing='original_RGB', mask_ratio=0.0, batch_size=32, normalization='L2',
             torch=PACKAGES['torch'], transformers=PACKAGES['transformers']), predict_embeddings)
finally:
    magi_model = None
    gc.collect()
    torch.cuda.empty_cache()

assert len(embeddings_per_page) == len(magi_raw_results)
for result, embeddings in zip(magi_raw_results, embeddings_per_page):
    assert len(embeddings) == len(result['characters']), 'Character/embedding count mismatch'
    assert np.isfinite(np.asarray(embeddings)).all(), 'Nonfinite embeddings'
per_page_results = copy.deepcopy(magi_raw_results)
print('Text counts:', [len(r['texts']) for r in per_page_results])


# --- Notebook inference cell 14 ---
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


def predict_paddle():
    output = [[] for _ in chapter_pages]
    if not any(result['texts'] for result in magi_raw_results):
        return output
    assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0, 'Paddle CUDA unavailable'
    assert PADDLE_DET_PATH.is_dir() and PADDLE_REC_PATH.is_dir(), 'Set the local Paddle model paths in cell 2.'
    # Explicit local directories prevent downloading detector/recognizer models.
    # Detector: https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
    # Recognizer: https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec
    engine = load_model_once('paddle', lambda: PaddleOCR(**PADDLE_SETTINGS, **PADDLE_LOCAL_SETTINGS))
    try:
        for page_index, (image, result) in enumerate(zip(chapter_pages, magi_raw_results)):
            for text_index, box in enumerate(result['texts']):
                crop, clipped = crop_box(image, box)
                record = dict(page_index=page_index, text_index=text_index, text_id=f'T{text_index+1}',
                              box=box, crop_box=clipped, text='', lines=[], status='invalid_crop', error=None)
                if crop is not None:
                    try:
                        for prediction in engine.predict(input=crop):
                            texts, scores = prediction['rec_texts'], prediction['rec_scores']
                            assert len(texts) == len(scores), 'Paddle text/score count mismatch'
                            for text, score in zip(texts, scores):
                                record['lines'].append(dict(text=str(text), confidence=float(score)))
                        record['text'] = ' '.join(line['text'] for line in record['lines'])
                        record['status'] = 'text' if record['text'].strip() else 'empty'
                    except Exception as error:
                        record.update(text='', lines=[], status='error', error=str(error))
                        print(f'PG{page_index+1}:T{text_index+1} OCR error: {error}')
                output[page_index].append(record)
    finally:
        del engine
        gc.collect()
        paddle.device.cuda.empty_cache()
    return output


# --- Notebook inference cell 16 ---
paddle_ocr_results = cached_stage('paddle_ocr',
    dict(magi=stage_ref('magi_raw'), paddle=PADDLE_SETTINGS, local_models=PADDLE_LOCAL_SETTINGS,
         packages={k: PACKAGES[k] for k in ['paddlepaddle-gpu', 'paddleocr', 'paddlex', 'numpy', 'pillow']}),
    predict_paddle)
print('Paddle records:', sum(len(page) for page in paddle_ocr_results))
print('Cached OCR errors:', sum(r['status'] == 'error' for page in paddle_ocr_results for r in page))


# --- Notebook inference cell 18 ---
SYSTEM_PROMPT = 'Classify the Paddle OCR text of one manga text box. Return exactly Text or Artifact. Text includes dialogue, narration, names, Japanese words/names, unfinished sentences, multiple sentences and multiple lines. Profanity and figurative language are allowed. Integers, integer-only text and numbers within meaningful dialogue are Text. Punctuation-only expressions are Text. Vocal reactions, screams, disgust, hesitation and stretched spoken words such as AAAAHH!!!, NOO!! and EWWW! are Text. Distinguish a human vocal reaction from a written environmental or action sound effect: Whoosh, BAM, CLANG and similar sound effects are Artifact. Artifact also includes nonsensical OCR noise, scan-group watermarks, website links and isolated code-like strings such as 2-B, 3-2 and V45S. Single Text Char are Artifacts Do not reject an unfamiliar name or a meaningful sentence simply because it contains a number or identifier. If uncertain, prefer Text. Do not correct, rewrite or explain the text.'

# One Paddle OCR input and one label per example.
EXAMPLES = [
    ('GAAAAAAHH!!!', 'Text'),
    ('AAAAHH!!!', 'Text'),
    ('AAAH!', 'Text'),
    ('NOO!!', 'Text'),
    ('EWWW!', 'Text'),
    ('UGH... THAT SMELLS AWFUL!', 'Text'),
    ('HUH?!', 'Text'),
    ('OH...', 'Text'),
    ('UM... I THINK SO.', 'Text'),
    ('HAHAHA!', 'Text'),
    ('OW! THAT HURTS!', 'Text'),
    ('YAAAY!!', 'Text'),
    ('I SAID TO NOT DO THAT.', 'Text'),
    ('But our characters were unaware', 'Text'),
    ('NANIKAWA', 'Text'),
    ('SHIROGANE, WAIT!', 'Text'),
    ("IT'S A GODDAMN MIRACLE, THE MONEY JUST VANISHED INTO THIN AIR.", 'Text'),
    ('IF SHE SUCCEEDS IN LURING OUT ONE PERSON IN THIS PLACE,', 'Text'),
    ('WAIT! WHERE DID EVERYONE GO? I WAS JUST HERE.', 'Text'),
    ('I TRUSTED YOU. NOW TELL ME WHAT HAPPENED.', 'Text'),
    ('WE HAVE TO LEAVE.\nTHE TRAIN IS ALREADY HERE!', 'Text'),
    ('Among the six who possessed the aptitude,\nSaiko-chan had the highest aptitude.', 'Text'),
    ('123', 'Text'),
    ('42', 'Text'),
    ('0', 'Text'),
    ('I HAVE 2 QUESTIONS.', 'Text'),
    ('WE HAVE 3 MINUTES LEFT!', 'Text'),
    ('HE PAID 500 YEN FOR IT.', 'Text'),
    ('I AM IN CLASS 2-B.', 'Text'),
    ('THE SCORE IS 3-2. WE CAN STILL WIN!', 'Text'),
    ('...', 'Text'),
    ('!!!!', 'Text'),
    ('...!!!', 'Text'),
    ('?!', 'Text'),
    ('もしもし', 'Artifact'),
    ('しし', 'Artifact'),
    ('も', 'Artifact'),
    ('a ~', 'Artifact'),
    ('~~~~~ ', 'Artifact'),
    ('~~~~~ ', 'Artifact'),
    ('Whoosh!!', 'Artifact'),
    ('VOOSH', 'Artifact'),
    ('KIRK', 'Artifact'),
    ('bEE', 'Artifact'),
    ('WFAG', 'Artifact'),
    ('VOOOSH', 'Artifact'),
    ('O', 'Artifact'),
    ('K', 'Artifact'),
    ('AKK', 'Artifact'),
    ('HUPP', 'Artifact'),
    ('CRASHHH', 'Artifact'),
    ('SHHH', 'Artifact'),
    ('KRASHHH', 'Artifact'),
    ('BAM', 'Artifact'),
    ('CLANG!', 'Artifact'),
    ('THUD', 'Artifact'),
    ('CRASH!!', 'Artifact'),
    ('RUSTLE RUSTLE', 'Artifact'),
    ('CLICK', 'Artifact'),
    ('SPLASH!', 'Artifact'),
    ('SFX', 'Artifact'),
    ('SKX', 'Artifact'),
    ('www.google.com', 'Artifact'),
    ('https://mangastream.com/read/123', 'Artifact'),
    ('MANGA STREAM DO NOT MIRROR', 'Artifact'),
    ('M4NGA STRE4M', 'Artifact'),
    ('DO NOT REDISTRIBUTE', 'Artifact'),
    ('oo', 'Artifact'),
    ('Chapter 32', 'Artifact'),
    ('Chapter 12', 'Artifact'),
    ('Chapter Over', 'Artifact'),
    ('Chapter End', 'Artifact'),
    ('To be Continued', 'Artifact'),
    ('Chapter', 'Artifact'),
    ('XZQWZXQ', 'Artifact'),
    ('2-B', 'Artifact'),
    ('3-2', 'Artifact'),
    ('V45S', 'Artifact'),
]


def punctuation_only(text):
    text=str(text or '').strip()
    return bool(text) and all(c.isspace() or unicodedata.category(c).startswith('P') for c in text)


def direct_decision(paddle,magi=''):
    # Only punctuation may be recovered from Magi when Paddle finds no text.
    if not str(paddle or '').strip():
        if punctuation_only(magi):return 'Text','Magi punctuation fallback: Paddle OCR empty'
        return 'Artifact','Paddle OCR empty; no Magi punctuation fallback'
    if punctuation_only(paddle):return 'Text','punctuation-only override'
    return None,None


def paddle_prompt(paddle):
    return json.dumps({'Paddle OCR':str(paddle or '')},ensure_ascii=False)


def classify_paddle(paddle,model,tokenizer):
    messages=[dict(role='system',content=SYSTEM_PROMPT)]
    for example,label in EXAMPLES:
        messages.extend([dict(role='user',content=paddle_prompt(example)),dict(role='assistant',content=label)])
    messages.append(dict(role='user',content=paddle_prompt(paddle)))
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=8)
    answer=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    # Unexpected model output stays visible as Review instead of causing deletion.
    label={'text':'Text','artifact':'Artifact'}.get(answer.casefold(),'Review')
    return dict(label=label,raw_answer=answer,reason='Gemma classification' if label!='Review' else 'unexpected Gemma answer')


# Keep original indices/boxes for character associations; choose Paddle or Magi punctuation only.
def select_sequence_ocr():
    selections=[]
    for page_index,(raw,paddle_records) in enumerate(zip(magi_raw_results,paddle_ocr_results)):
        assert len(raw['texts'])==len(raw['ocr'])==len(paddle_records)
        page=[]
        for text_index,(box,magi_text,record) in enumerate(zip(raw['texts'],raw['ocr'],paddle_records)):
            assert record['box']==box and record['text_index']==text_index and record['page_index']==page_index
            paddle_text=str(record['text'] or '');magi_text=str(magi_text or '')
            text=paddle_text.strip() if paddle_text.strip() else magi_text.strip() if punctuation_only(magi_text) else ''
            source='paddle' if paddle_text.strip() else 'magi_punctuation' if text else None
            page.append(dict(page_index=page_index,text_index=text_index,text_id=f'T{text_index+1}',box=box,
                magi_text=magi_text,paddle_text=paddle_text,selected_text=text,selected_source=source,
                reason='paddle' if source=='paddle' else 'Magi punctuation fallback' if text else 'empty Paddle; no punctuation fallback',
                needs_review=False,disagreement=magi_text.casefold()!=paddle_text.casefold(),
                paddle_status=record['status'],paddle_error=record['error'],paddle_lines=record['lines']))
        selections.append(page)
    return selections


def classify_artifact_sequence():
    by_text={};pending=[]
    for page in ocr_selection:
        for record in page:
            label,reason=direct_decision(record['paddle_text'],record['magi_text'])
            key=record['paddle_text']
            if label is None and key not in pending:pending.append(key)
    if pending:
        from transformers import AutoTokenizer,Gemma3ForConditionalGeneration,BitsAndBytesConfig
        assert torch.cuda.is_available(),'Gemma requires the CUDA godfather.venv kernel'
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Model/tokenizer: https://huggingface.co/google/gemma-3-4b-it
        tokenizer=load_model_once('gemma_tokenizer', lambda: AutoTokenizer.from_pretrained(str(GEMMA_MODEL_PATH),local_files_only=True))
        model=load_model_once('gemma', lambda: Gemma3ForConditionalGeneration.from_pretrained(str(GEMMA_MODEL_PATH),local_files_only=True,
            device_map='auto',dtype=dtype,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype)).eval())
        try:
            for i,text in enumerate(pending,1):
                by_text[text]=classify_paddle(text,model,tokenizer)
                print('Artifact classification',i,'/',len(pending),by_text[text]['label'])
        finally:
            del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    pages=[]
    for records in ocr_selection:
        page=[]
        for record in records:
            label,reason=direct_decision(record['paddle_text'],record['magi_text'])
            decision=dict(label=label,reason=reason,raw_answer='') if label else by_text[record['paddle_text']]
            a=''.join(re.findall('[a-z0-9]',record['magi_text'].casefold()))
            b=''.join(re.findall('[a-z0-9]',record['paddle_text'].casefold()))
            page.append(dict(gemma_label=decision['label'],artifact_label=decision['label'],
                gemma_raw_answer=decision['raw_answer'],artifact_reason=decision['reason'],
                ocr_agreement=SequenceMatcher(None,a,b).ratio() if a and b else None,
                delete_ocr=decision['label']=='Artifact',
                deletion_reason=decision['reason'] if decision['label']=='Artifact' else '',
                punctuation_bubble_checked=False,punctuation_bubble_found=None))
        pages.append(page)
    return pages


# --- Notebook inference cell 20 ---
ocr_selection=cached_stage('ocr_selection',dict(magi=stage_ref('magi_raw'),paddle=stage_ref('paddle_ocr'),
    algorithm=function_digest(select_sequence_ocr),punctuation=function_digest(punctuation_only)),select_sequence_ocr)
artifact_initial=cached_stage('artifact_labels',dict(ocr=stage_ref('ocr_selection'),model=GEMMA_IDENTITY,
    system=SYSTEM_PROMPT,examples=EXAMPLES,input='paddle_only',quantization='nf4',max_new_tokens=8,
    transformers=version('transformers'),bitsandbytes=version('bitsandbytes'),
    algorithm=function_digest(classify_artifact_sequence),classifier=function_digest(classify_paddle),
    direct=function_digest(direct_decision)),classify_artifact_sequence)
# This adapter preserves the existing character/speaker mapping contract.
ocr_agreement=copy.deepcopy(artifact_initial)  # Agreement is diagnostic only; it never deletes here.
per_page_results=copy.deepcopy(magi_raw_results)
for page_index,(result,decisions,labels) in enumerate(zip(per_page_results,ocr_selection,artifact_initial)):
    result['magi_ocr']=list(result['ocr'])
    result['ocr']=[d['selected_text'] for d in decisions]
    for decision,label in zip(decisions,labels):decision['needs_review']=label['gemma_label']=='Review'
display(pd.DataFrame([dict(page=PAGE_LABELS[p],box_id=f"PG{PAGE_LABELS[p]}:T{i+1}",
    paddle=d['paddle_text'],magi=d['magi_text'],text=d['selected_text'],**artifact_initial[p][i])
    for p,page in enumerate(ocr_selection) for i,d in enumerate(page)]))


# --- Notebook inference cell 22 ---
def remove_duplicate_characters(results, embeddings, min_iou=0.90, min_cosine=0.95):
    """Remove contained, visually similar duplicates; preserve all text indices."""
    assert 0 <= min_iou <= 1 and -1 <= min_cosine <= 1
    assert len(results) == len(embeddings)
    cleaned_pages, cleaned_embeddings, audits = [], [], []
    for page_index, (raw, vectors) in enumerate(zip(results, embeddings)):
        boxes = np.asarray(raw['characters'], dtype=float).reshape(-1, 4)
        assert len(vectors) == len(boxes), 'Character/embedding count mismatch'
        assert len(raw['character_cluster_labels']) == len(boxes)
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        # Largest first. Equal areas prefer the earlier original character index.
        retained, representatives, decisions = [], {}, []
        for inner in sorted(range(len(boxes)), key=lambda i: (-areas[i], i)):
            representative = inner
            for outer in retained:
                a, b = boxes[inner], boxes[outer]
                if not np.isfinite(a).all() or not np.isfinite(b).all() or areas[inner] <= 0 or areas[outer] <= 0:
                    continue
                contained = a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
                iou = float(areas[inner] / areas[outer]) if contained else 0.0
                if not contained or iou < min_iou:
                    continue
                x, y = np.asarray(vectors[inner], dtype=float), np.asarray(vectors[outer], dtype=float)
                assert x.ndim == y.ndim == 1 and x.shape == y.shape
                denominator = np.linalg.norm(x) * np.linalg.norm(y)
                if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(denominator) or denominator <= 0:
                    continue
                cosine = float(np.clip(np.dot(x, y) / denominator, -1, 1))
                if cosine >= min_cosine:
                    representative = outer
                    decisions.append(dict(removed_original_index=inner, retained_original_index=outer,
                                          smaller_box_coverage=1.0, iou=iou, cosine=cosine))
                    break
            representatives[inner] = representative
            if representative == inner:
                retained.append(inner)
        # Keep the original relative order of surviving character appearances.
        kept = sorted(retained)
        new_index = {old: new for new, old in enumerate(kept)}
        old_to_new = [new_index[representatives[i]] for i in range(len(boxes))]
        page = copy.deepcopy(raw)
        for key in ['characters', 'character_cluster_labels', 'character_names']:
            if key in raw:
                assert len(raw[key]) == len(boxes), f'Misaligned {key}'
                page[key] = [raw[key][i] for i in kept]
        links, seen = [], set()
        for text_index, char_index in raw['text_character_associations']:
            assert 0 <= text_index < len(raw['texts']) and 0 <= char_index < len(boxes)
            link = (text_index, old_to_new[char_index])
            if link not in seen:
                seen.add(link)
                links.append(list(link))
        page['text_character_associations'] = links
        assert page['texts'] == raw['texts'] and page['ocr'] == raw['ocr']
        cleaned_pages.append(page)
        cleaned_embeddings.append([vectors[i] for i in kept])
        audits.append(dict(page_index=page_index, kept_original_indices=kept,
                           original_to_cleaned_index=old_to_new, removals=decisions))
    return dict(pages=cleaned_pages, embeddings=cleaned_embeddings, audit=audits)


character_cleanup = cached_stage('character_cleanup',
    dict(magi=stage_ref('magi_raw'), embeddings=stage_ref('embeddings'),
         cleanup=CHARACTER_CLEANUP_SETTINGS),
    lambda: remove_duplicate_characters(magi_raw_results, embeddings_per_page,
        CHARACTER_CLEANUP_SETTINGS['min_iou'], CHARACTER_CLEANUP_SETTINGS['min_cosine']))
character_results = character_cleanup['pages']
character_embeddings_per_page = character_cleanup['embeddings']
validate_magi(character_results)
# Apply only character fields; preserve the selected OCR already in memory.
for result, cleaned in zip(per_page_results, character_results):
    for key in ['characters', 'character_cluster_labels', 'character_names', 'text_character_associations']:
        if key in cleaned:
            result[key] = copy.deepcopy(cleaned[key])
print('Character counts before:', [len(r['characters']) for r in magi_raw_results])
print('Character counts after: ', [len(r['characters']) for r in character_results])
display(pd.DataFrame([dict(page_index=page['page_index'], **decision)
                      for page in character_cleanup['audit'] for decision in page['removals']]))


# --- Notebook inference cell 24 ---
def page_clusters(result, page_index):
    groups = {}
    for char_index, label in enumerate(result['character_cluster_labels']):
        cluster_id = f'PG{page_index+1}:CL{int(label)+1}' if int(label) >= 0 else f'PG{page_index+1}:C{char_index+1}:unassigned'
        groups.setdefault(cluster_id, []).append(char_index)
    return groups


def clusterize(results, embeddings, threshold=0.50):
    assert -1 <= threshold <= 1, 'Cosine threshold must be between -1 and 1'
    vectors = [np.asarray(page, dtype=float) for page in embeddings]
    assert len(results) == len(vectors)
    for result, page_vectors in zip(results, vectors):
        assert len(result['characters']) == len(page_vectors), 'Embedding count mismatch'
    entities, decisions, char_to_entity = {}, [], {}
    for page_index, result in enumerate(results):
        clusters = page_clusters(result, page_index)
        candidates = []
        for cluster_id, indices in clusters.items():
            for entity_id, entity in entities.items():
                best_score, best_pair = -float('inf'), None
                for current in indices:
                    for member in entity['members']:
                        score = float(np.dot(vectors[page_index][current], vectors[member['page_index']][member['character_index']]))
                        if score > best_score:
                            best_score = score
                            best_pair = dict(current=f'PG{page_index+1}:C{current+1}', existing=member['character_id'])
                if best_score >= threshold:
                    candidates.append(dict(cluster_id=cluster_id, entity_id=entity_id, score=best_score, best_pair=best_pair))
        candidates.sort(key=lambda candidate: candidate['score'], reverse=True)
        accepted, used_entities = {}, set()
        for candidate in candidates:
            if candidate['cluster_id'] not in accepted and candidate['entity_id'] not in used_entities:
                accepted[candidate['cluster_id']] = candidate
                used_entities.add(candidate['entity_id'])
        for cluster_id, indices in clusters.items():
            match = accepted.get(cluster_id)
            entity_id = match['entity_id'] if match else f'E{len(entities)+1}'
            if entity_id not in entities:
                entities[entity_id] = dict(clusters=[], members=[])
            entities[entity_id]['clusters'].append(cluster_id)
            for char_index in indices:
                character_id = f'PG{page_index+1}:C{char_index+1}'
                entities[entity_id]['members'].append(dict(page_index=page_index, character_index=char_index,
                                                         character_id=character_id, cluster_id=cluster_id))
                char_to_entity[character_id] = entity_id
            decisions.append(dict(page_index=page_index, cluster_id=cluster_id, entity_id=entity_id,
                                  reason='matched' if match else 'new_entity',
                                  score=match['score'] if match else None, best_pair=match['best_pair'] if match else None))
    return dict(entities=entities, char_to_entity=char_to_entity, decisions=decisions)


# --- Notebook inference cell 26 ---
entity_results = cached_stage('entities',
    dict(characters=stage_ref('character_cleanup'), threshold=ENTITY_THRESHOLD,
         algorithm='sequential_best_member_one_cluster_per_entity_per_page'),
    lambda: clusterize(character_results, character_embeddings_per_page, ENTITY_THRESHOLD))
identified_char = entity_results['entities']
char_to_entity = entity_results['char_to_entity']
assert len(char_to_entity) == sum(len(r['characters']) for r in character_results)
print('Entities:', len(identified_char))
display(pd.DataFrame(entity_results['decisions']))


# --- Notebook inference cell 28 ---
def map_text_entities(results, selections, entity_map, sequence_id, page_names):
    assert len(results) == len(selections) == len(page_names)
    records = []
    for page_index, (result, decisions) in enumerate(zip(results, selections)):
        assert len(result['texts']) == len(result['ocr']) == len(decisions)
        links_by_text = {}
        for text_index, char_index in result['text_character_associations']:
            assert 0 <= text_index < len(result['texts']) and 0 <= char_index < len(result['characters'])
            links_by_text.setdefault(text_index, []).append(char_index)
        for text_index, (box, text, decision) in enumerate(zip(result['texts'], result['ocr'], decisions)):
            assert decision['page_index'] == page_index and decision['text_index'] == text_index and decision['box'] == box
            assert decision['selected_text'] == text
            links = []
            for char_index in dict.fromkeys(links_by_text.get(text_index, [])):
                char_id = f'PG{page_index+1}:C{char_index+1}'
                links.append(dict(character_index=char_index, character_id=char_id, entity_id=entity_map[char_id]))
            entity_ids = list(dict.fromkeys(link['entity_id'] for link in links))
            records.append(dict(sequence_id=sequence_id, page_index=page_index, page_file=page_names[page_index],
                                text_index=text_index, text_id=f'T{text_index+1}', box=box, text=text,
                                entity_id=entity_ids[0] if len(entity_ids) == 1 else None,
                                entity_ids=entity_ids, character_links=links,
                                association_status='unassociated' if not links else 'linked' if len(entity_ids) == 1 else 'ambiguous',
                                ocr_source=decision['selected_source'], ocr_reason=decision['reason'],
                                needs_review=decision['needs_review'] or len(entity_ids) > 1,
                                disagreement=decision['disagreement']))
    return records


# --- Notebook inference cell 30 ---
text_entities = cached_stage('text_entities',
    dict(magi=stage_ref('magi_raw'), characters=stage_ref('character_cleanup'),
         selector=stage_ref('ocr_selection'), artifacts=stage_ref('artifact_labels'),entities=stage_ref('entities')),
    lambda: map_text_entities(per_page_results, ocr_selection, char_to_entity, SEQUENCE_ID,
                             [path.name for path in PAGE_FILES]))
assert len(text_entities) == sum(len(r['texts']) for r in magi_raw_results)
columns = ['page_index', 'text_id', 'text', 'entity_id', 'association_status',
           'ocr_source', 'needs_review', 'disagreement']
display(pd.DataFrame(text_entities, columns=columns))

for record in text_entities:
    record['box_id']=f"PG{PAGE_LABELS[record['page_index']]}:T{record['text_index']+1}"
    record.update(ocr_agreement[record['page_index']][record['text_index']])
    decision=ocr_selection[record['page_index']][record['text_index']]
    record.update(magi_text=decision['magi_text'],paddle_text=decision['paddle_text'])
# Keep these candidates immutable; final group speakers are resolved later.
raw_text_entities=copy.deepcopy(text_entities)


# --- Notebook inference cell 32 ---
BUBBLE_SHA256=file_hash(BUBBLE_MODEL)


def predict_bubbles():
    # No external cache discovery: cached_stage reads only this notebook's own cache.
    from ultralytics import YOLO
    # Weights: https://huggingface.co/PSImera/manga_bubbles_detect
    detector=load_model_once('yolo', lambda: YOLO(str(BUBBLE_MODEL)))
    pages=[]
    try:
        for image in crop_pages:
            prediction=detector.predict(source=Image.fromarray(image),imgsz=DETECTION['imgsz'],
                conf=DETECTION['confidence'],device=0 if DEVICE=='cuda' else 'cpu',verbose=False)[0]
            height,width=image.shape[:2];candidates=[]
            for box,score in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist()):
                a,b,c,d=box
                bounds=[max(0,math.floor(a)),max(0,math.floor(b)),min(width,math.ceil(c)),min(height,math.ceil(d))]
                if bounds[2]>bounds[0] and bounds[3]>bounds[1]:candidates.append(dict(box=bounds,confidence=float(score)))
            candidates.sort(key=lambda b:(b['box'][1],-b['box'][0]))
            pages.append(dict(candidates=candidates,source='own_local_yolo'))
            del prediction
    finally:
        del detector;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return pages


bubble_pages=cached_stage('contour_bubbles',dict(weights_sha256=BUBBLE_SHA256,settings=DETECTION,
    ultralytics=version('ultralytics'),image_mode='original_RGB',algorithm=function_digest(predict_bubbles)),predict_bubbles)
print('Bubble candidates:',[len(p['candidates']) for p in bubble_pages])


# --- Notebook inference cell 34 ---
CONTOUR_INFO=dict(module_sha256=file_hash(CONTOUR_MODULE_PATH),
    grouping_sha256=file_hash(CONTOUR_GROUPING_PATH),check=CONTOUR_SETTINGS,
    region=REGION_SETTINGS,faint_region=FAINT_REGION_SETTINGS,group_faint=GROUP_FAINT_SETTINGS)


def filter_artifact_punctuation():
    filtered=copy.deepcopy(artifact_initial)
    for page_index,(selections,labels) in enumerate(zip(ocr_selection,filtered)):
        text_boxes=[d['box'] for d in selections]
        image=Image.fromarray(crop_pages[page_index])
        for index,(decision,label) in enumerate(zip(selections,labels)):
            if label['delete_ocr'] or decision['paddle_text'].strip() or not punctuation_only(decision['magi_text']):continue
            found=False;checked_candidate=None
            a,b,c,d=decision['box'];area=max(1,(c-a)*(d-b))
            for candidate_id,candidate in enumerate(bubble_pages[page_index]['candidates']):
                if contour_model.contours.overlap(decision['box'],candidate['box'])/area<.8:continue
                result=contour_model.find_background_regions(candidate,image,text_boxes,index)
                region=result['best']
                if (region is not None and not region['leak_suspected']
                    and region['visible_points']>=REGION_SETTINGS['min_visible_points']):
                    found=True;checked_candidate=candidate_id;break
            label.update(punctuation_bubble_checked=True,punctuation_bubble_found=found,
                punctuation_candidate_id=checked_candidate)
            if not found:
                label.update(artifact_label='Artifact',delete_ocr=True,
                    deletion_reason='Empty Paddle + Magi punctuation has no qualifying bubble mask')
    return filtered


ocr_agreement=cached_stage('artifact_filter',dict(labels=stage_ref('artifact_labels'),
    bubbles=stage_ref('contour_bubbles'),contour=CONTOUR_INFO,coverage=.8,
    algorithm=function_digest(filter_artifact_punctuation)),filter_artifact_punctuation)
# The same final deletion flags feed grouping, association routing and all exports.
for record in text_entities:
    record.update(ocr_agreement[record['page_index']][record['text_index']])
raw_text_entities=copy.deepcopy(text_entities)


def contour_texts(page_index):
    texts=[];page_number=PAGE_LABELS[page_index]
    for index,decision in enumerate(ocr_selection[page_index]):
        filtered=ocr_agreement[page_index][index]
        if filtered['delete_ocr']:continue
        texts.append(dict(box_id=f'PG{page_number}:T{index+1}',text_index=index,box=decision['box'],
            text=decision['selected_text'],magi_text=decision['magi_text'],paddle_text=decision['paddle_text'],
            ocr_agreement=filtered['ocr_agreement']))
    return texts


def make_contour_roles():
    pages=[]
    for page_index,path in enumerate(PAGE_FILES):
        analysis,_=contour_model.analyze_page(bubble_pages[page_index]['candidates'],
            Image.fromarray(crop_pages[page_index]),contour_texts(page_index),PAGE_LABELS[page_index])
        for row in analysis['rows']:
            row.update(sequence_id=SEQUENCE_ID,page_index=page_index,page_file=path.name)
            row['contour_text_role']=row['text_role'];row['contour_original_reason']=row['reason']
            member_ids=set(row['box_ids']);tail_ids=[];tail_members=[]
            raw=magi_raw_results[page_index]
            for text_index,tail_index in raw['text_tail_associations']:
                assert 0<=text_index<len(raw['texts']) and 0<=tail_index<len(raw['tails'])
                bid=f'PG{PAGE_LABELS[page_index]}:T{text_index+1}'
                if bid in member_ids:tail_ids.append(tail_index);tail_members.append(bid)
            row['tail_ids']=sorted(set(tail_ids));row['tail_member_box_ids']=sorted(set(tail_members))
            row['tail_override']=bool(tail_ids)
            if row['tail_override']:
                row['text_role']='SPOKEN';row['reason']='Magi text-to-tail association forces SPOKEN'
        pages.append(analysis)
    return pages


contour_pages=cached_stage('contour_roles',
    dict(magi=stage_ref('magi_raw'),selector=stage_ref('ocr_selection'),filter=stage_ref('artifact_filter'),
         bubbles=stage_ref('contour_bubbles'),contour=CONTOUR_INFO,
         algorithm=function_digest(make_contour_roles),adapter=function_digest(contour_texts)),make_contour_roles)
contour_rows=[r for page in contour_pages for r in page['rows']]
expected={r['box_id'] for r in raw_text_entities if not r['delete_ocr']}
emitted=[bid for row in contour_rows for bid in row['box_ids']]
assert len(emitted)==len(set(emitted)) and set(emitted)==expected,'Missing/duplicated surviving text'
display(pd.DataFrame(contour_rows).reindex(columns=['box_id','box_ids','text','text_role','reason','needs_review']))


# --- Notebook inference cell 36 ---
def resolve_group_speaker(role,members):
    entity_ids=list(dict.fromkeys(entity for member in members for entity in member['entity_ids']))
    missing=[m['box_id'] for m in members if not m['entity_ids']]
    links=[dict(box_id=m['box_id'],entity_ids=m['entity_ids'],character_links=m['character_links'],
                association_status=m['association_status']) for m in members]
    if role=='NARRATED':speaker,status,review='NARRATION','narrator',False
    elif len(entity_ids)>1:speaker,status,review=None,'conflicting_entities',True
    elif not entity_ids:speaker,status,review=None,'unassociated',True
    elif missing:speaker,status,review=entity_ids[0],'partially_linked',True
    else:speaker,status,review=entity_ids[0],'linked',False
    return dict(speaker=speaker,speaker_status=status,speaker_needs_review=review,
                candidate_entity_ids=entity_ids,unlinked_member_ids=missing,member_associations=links)


def make_group_speakers():
    lookup={r['box_id']:r for r in raw_text_entities if not r['delete_ocr']}
    rows=[]
    for contour in contour_rows:
        members=[lookup[bid] for bid in contour['box_ids']]
        assert all(m['page_index']==contour['page_index'] for m in members)
        row=copy.deepcopy(contour)
        row.update(resolve_group_speaker(row['text_role'],members))
        ocr_review=any(ocr_selection[m['page_index']][m['text_index']]['needs_review'] for m in members)
        row['needs_review']=bool(row['needs_review'] or row['speaker_needs_review'] or ocr_review)
        row['ocr_needs_review']=ocr_review
        rows.append(row)
    return rows


grouped_records=cached_stage('contour_speakers',
    dict(roles=stage_ref('contour_roles'),associations=stage_ref('text_entities'),filter=stage_ref('artifact_filter'),
         algorithm=function_digest(make_group_speakers),routing=function_digest(resolve_group_speaker)),make_group_speakers)
# Export an audit of all original boxes, including deleted boxes, without reordering them.
by_member={bid:row for row in grouped_records for bid in row['box_ids']}
box_records=[]
for original in raw_text_entities:
    row=copy.deepcopy(original);group=by_member.get(row['box_id'])
    if row['delete_ocr']:
        assert group is None
        row.update(group_id=None,text_role=None,speaker=None,speaker_status='deleted')
    else:
        assert group is not None
        row.update(group_id=group['box_id'],text_role=group['text_role'],speaker=group['speaker'],
            speaker_status=group['speaker_status'],needs_review=group['needs_review'],contour_reason=group['reason'],
            tail_override=group['tail_override'],tail_ids=group['tail_ids'],
            contour_text_role=group['contour_text_role'])
    box_records.append(row)
display(pd.DataFrame(grouped_records).reindex(columns=['box_id','box_ids','speaker','speaker_status','text_role','text','needs_review']))


# --- Notebook inference cell 42 ---
def build_transcript(sequence_id,records,page_files,sequence_files,entity_ids):
    # Stable sequence-local names; these are predicted identities, not GT character matches.
    ordered_entities=sorted(entity_ids,key=lambda eid:(int(eid[1:]) if eid.startswith('E') and eid[1:].isdigit() else float('inf'),eid))
    names={entity:f'character{i+1}' for i,entity in enumerate(ordered_entities)}
    page_positions={str(path.resolve()):i for i,path in enumerate(sequence_files)}
    pages=[[] for _ in sequence_files]
    for row in sorted(records,key=lambda r:(r['page_index'],r['text_index'])):
        page=page_positions[str(page_files[row['page_index']].resolve())]
        speaker='NARRATION' if row['text_role']=='NARRATED' else names.get(row['speaker'],'UNKNOWN')
        pages[page].append(dict(speaker=speaker,text=row['text']))
    return dict(sequence_id=sequence_id,pages=pages),names


# Keep detailed exports; the transcript below has exactly the labels.jsonl contract.
# sequence_files is the selected explicit image list or original dataset page list.
transcript,entity_to_character=build_transcript(SEQUENCE_ID,grouped_records,PAGE_FILES,sequence_files,identified_char)
write_json(OUTPUT_DIR/'transcript.json',transcript)
# One sequence object per physical line. Replace this run's file rather than appending duplicates.
(OUTPUT_DIR/'transcript.jsonl').write_text(json.dumps(transcript,ensure_ascii=False)+'\n',encoding='utf-8')
write_json(OUTPUT_DIR/'transcript_speaker_mapping.json',entity_to_character)


dialogue_pages=[[] for _ in PAGE_FILES]
for row in grouped_records:
    dialogue_pages[row['page_index']].append(dict(group_id=row['box_id'],box_ids=row['box_ids'],
        speaker=row['speaker'],speaker_status=row['speaker_status'],text_role=row['text_role'],
        text=row['text'],needs_review=row['needs_review']))
write_json(OUTPUT_DIR/'dialogue_pages.json',dialogue_pages)
write_json(OUTPUT_DIR/'group_predictions.json',grouped_records)
pd.DataFrame(grouped_records).to_csv(OUTPUT_DIR/'group_predictions.csv',index=False)
write_json(OUTPUT_DIR/'box_predictions.json',box_records)
pd.DataFrame(box_records).to_csv(OUTPUT_DIR/'box_predictions.csv',index=False)
deleted=[r for r in box_records if r['delete_ocr']]
write_json(OUTPUT_DIR/'deleted_ocr.json',deleted)
pd.DataFrame(deleted).to_csv(OUTPUT_DIR/'deleted_ocr.csv',index=False)
write_json(OUTPUT_DIR/'grouping_audit.json',[p['grouping'] for p in contour_pages])
write_json(OUTPUT_DIR/'entities.json',entity_results)
write_json(OUTPUT_DIR/'run_metadata.json',dict(sequence_id=SEQUENCE_ID,pages=PAGE_METADATA,
    narration_method='contour_narrator2',contour=CONTOUR_INFO,bubble_weights_sha256=BUBBLE_SHA256,
    bubble_settings=DETECTION,character_checkpoint=CHECKPOINT_INFO,entity_threshold=ENTITY_THRESHOLD,
    association_head=str(ASSOCIATION_HEAD_PATH) if ASSOCIATION_HEAD_PATH else 'original_magi',
    tail_override=True,ocr_filter=dict(method='Paddle-only local Gemma',
        agreement_deletion=False,punctuation_fallback_requires_bubble=True),dependencies=RESOURCE_MANIFEST,
    manifest=str(MANIFEST_PATH),stages=manifest['stages']))
lines=[]
for path,page in zip(sequence_files,transcript['pages']):
    lines.append(f'[Page {path.name}]')
    lines.extend(f"{row['speaker']}: {row['text']}" for row in page)
    lines.append('')
(OUTPUT_DIR/'transcript.txt').write_text('\n'.join(lines),encoding='utf-8')

display(pd.DataFrame([dict(page=path.name,detected=len(magi_raw_results[i]['texts']),
    deleted=sum(r['delete_ocr'] for r in box_records if r['page_index']==i),
    output_rows=len(dialogue_pages[i]),spoken=sum(r['text_role']=='SPOKEN' for r in dialogue_pages[i]),
    narrated=sum(r['text_role']=='NARRATED' for r in dialogue_pages[i]),
    review=sum(r['needs_review'] for r in dialogue_pages[i])) for i,path in enumerate(PAGE_FILES)]))
print('Saved:',OUTPUT_DIR)
# Optional annotated page export (no model work):
# for i,path in enumerate(PAGE_FILES):
#     figure=show_page(i,save_path=OUTPUT_DIR/f'annotated_{path.stem}.png')
#     plt.close(figure)

write_json(OUTPUT_DIR/'artifact_decisions.json',artifact_initial)
write_json(OUTPUT_DIR/'artifact_filter_audit.json',ocr_agreement)


# --- Notebook inference cell 44 ---
# This final stage changes transcript text only, after grouping and speaker assignment.
GEMMA_EXAMPLES=[
    {'role': 'system', 'content':
     'You restore English manga dialogue corrupted by OCR. '
     'Repair obvious broken words. Preserve legitimate names, valid single-letter words such as I and A, '
     'Preserve Punctutaion and Punctuation Only Sentences'
     'Remove nonsensical inserted letter sequences and accidental isolated letters that interrupt a sentence. '
     'capitalization, meaning, punctuation and unfinished sentences. '
     'Do not rewrite the dialogue, complete it or answer it. '
     'Preserve numbers, amounts, times and units; never delete a number while repairing nearby words. '
     'Repair obvious O/0 OCR confusion inside a numeric amount, such as 17O GRAND to 170 GRAND. '
     'any single character repeated multiple times could be punctuations mis interpreted '
     'Return only the restored text, with no explanation or quotation marks.'},
    {'role': 'user', 'content': 'PLEASE Q OPEN THE WINDOW.'},
    {'role': 'assistant', 'content': 'PLEASE OPEN THE WINDOW.'},
    {'role': 'user', 'content': 'I CAN HEAR XZQWZXQ SOMEONE OUTSIDE...'},
    {'role': 'assistant', 'content': 'I CAN HEAR SOMEONE OUTSIDE...'},
    {'role': 'user', 'content': 'AH... SHIROGANE, WAIT!'},
    {'role': 'assistant', 'content': 'AH... SHIROGANE, WAIT!'},
    {'role': 'user', 'content': '.....'},
    {'role': 'assistant', 'content': '.....'},
    {'role': 'user', 'content': 'PLEASE S OPEN THE WINDOW.'},
    {'role': 'assistant', 'content': 'PLEASE OPEN THE WINDOW.'},
    # Repair OCR words and numeric confusions without losing amounts or times.
    {'role': 'user', 'content': "I'M TAKING OHT 17O GRAND tO COVER YOUR LOAN AND INTEREST PAYMENTS."},
    {'role': 'assistant', 'content': "I'M TAKING OUT 170 GRAND TO COVER YOUR LOAN AND INTEREST PAYMENTS."},
    {'role': 'user', 'content': 'im worlkng in 20 minutes'},
    {'role': 'assistant', 'content': 'im working in 20 minutes'},
    {'role': 'user', 'content': 'WE HAVE 3 MINUTFS LEFT!'},
    {'role': 'assistant', 'content': 'WE HAVE 3 MINUTES LEFT!'},
    {'role': 'user', 'content': 'HE PA1D 500 YEN FOR IT.'},
    {'role': 'assistant', 'content': 'HE PAID 500 YEN FOR IT.'},
    {'role': 'user', 'content': 'I OWE YOU 170 GRAND.'},
    {'role': 'assistant', 'content': 'I OWE YOU 170 GRAND.'},
]
FORCE.setdefault('gemma_transcript',False)
STAGE_VERSIONS['gemma_transcript']=1
# Always rebuild from original group OCR, so rerunning cleanup cannot repeatedly edit its own output.
raw_transcript,_=build_transcript(SEQUENCE_ID,grouped_records,PAGE_FILES,sequence_files,identified_char)
# Keep the exact original OCR transcript before changing any final transcript exports.
write_json(OUTPUT_DIR/'transcript_raw.json',raw_transcript)
(OUTPUT_DIR/'transcript_raw.jsonl').write_text(json.dumps(raw_transcript,ensure_ascii=False)+'\n',encoding='utf-8')
raw_lines=[]
for path,page in zip(sequence_files,raw_transcript['pages']):
    raw_lines.append(f'[Page {path.name}]')
    raw_lines.extend(f"{row['speaker']}: {row['text']}" for row in page)
    raw_lines.append('')
(OUTPUT_DIR/'transcript_raw.txt').write_text('\n'.join(raw_lines),encoding='utf-8')
assert (GEMMA_MODEL_PATH/'config.json').is_file(),f'Missing local Gemma: {GEMMA_MODEL_PATH}'
GEMMA_IDENTITY=dict(path=str(GEMMA_MODEL_PATH),
    files={path.name:dict(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                         sha256=file_hash(path) if path.suffix=='.json' else None)
           for path in sorted(GEMMA_MODEL_PATH.iterdir())
           if path.is_file() and path.suffix in {'.json','.safetensors','.bin','.model'}})


def correct_transcript_text(text,model,tokenizer):
    if not text.strip() or punctuation_only(text):return text
    messages=GEMMA_EXAMPLES+[{'role':'user','content':text}]
    inputs=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,
            max_new_tokens=max(256,inputs['input_ids'].shape[1]*2))
    return tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()


def make_gemma_transcript():
    cleaned=copy.deepcopy(raw_transcript);audit=[]
    if not CLEAN_TRANSCRIPT or not any(row['text'].strip() and not punctuation_only(row['text']) for page in cleaned['pages'] for row in page):
        return dict(transcript=cleaned,audit=audit)
    from transformers import AutoTokenizer,Gemma3ForConditionalGeneration,BitsAndBytesConfig
    assert torch.cuda.is_available(),'Gemma cleanup requires your CUDA godfather.venv kernel.'
    compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Model/tokenizer: https://huggingface.co/google/gemma-3-4b-it
    tokenizer=load_model_once('gemma_tokenizer', lambda: AutoTokenizer.from_pretrained(str(GEMMA_MODEL_PATH),local_files_only=True))
    model=load_model_once('gemma', lambda: Gemma3ForConditionalGeneration.from_pretrained(str(GEMMA_MODEL_PATH),local_files_only=True,
        device_map='auto',dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                             bnb_4bit_compute_dtype=compute_dtype)).eval())
    try:
        # One transcript entry at a time; never send speaker IDs to the correction model.
        seen={}
        for page_index,page in enumerate(cleaned['pages']):
            for row_index,row in enumerate(page):
                original=row['text']
                if original not in seen:seen[original]=correct_transcript_text(original,model,tokenizer)
                answer=seen[original]
                row['text']=answer if answer else original
                audit.append(dict(page_index=page_index,row_index=row_index,speaker=row['speaker'],
                    raw_text=original,cleaned_text=row['text'],changed=row['text']!=original,
                    empty_output_fallback=bool(original.strip() and not answer)))
    finally:
        del model,tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return dict(transcript=cleaned,audit=audit)


# A compatible cache skips Gemma loading and generation on subsequent runs.
gemma_cleanup=cached_stage('gemma_transcript',dict(input_transcript=digest(raw_transcript),enabled=CLEAN_TRANSCRIPT,
    model=GEMMA_IDENTITY,prompt=GEMMA_EXAMPLES,quantization='nf4',generation='greedy_dynamic_max_tokens_v1',
    transformers=version('transformers'),bitsandbytes=version('bitsandbytes'),
    algorithm=function_digest(make_gemma_transcript),corrector=function_digest(correct_transcript_text)),make_gemma_transcript)
cleaned_transcript=gemma_cleanup['transcript']
assert cleaned_transcript['sequence_id']==raw_transcript['sequence_id']
assert len(cleaned_transcript['pages'])==len(raw_transcript['pages'])
for original_page,cleaned_page in zip(raw_transcript['pages'],cleaned_transcript['pages']):
    assert len(original_page)==len(cleaned_page)
    assert [r['speaker'] for r in original_page]==[r['speaker'] for r in cleaned_page]
    assert all(set(r)=={'speaker','text'} for r in cleaned_page)
# Final transcript filenames now contain the cleaned text in the original labels.jsonl schema.
transcript=copy.deepcopy(cleaned_transcript)
write_json(OUTPUT_DIR/'transcript.json',transcript)
(OUTPUT_DIR/'transcript.jsonl').write_text(json.dumps(transcript,ensure_ascii=False)+'\n',encoding='utf-8')
lines=[]
for path,page in zip(sequence_files,transcript['pages']):
    lines.append(f'[Page {path.name}]')
    lines.extend(f"{row['speaker']}: {row['text']}" for row in page)
    lines.append('')
(OUTPUT_DIR/'transcript.txt').write_text('\n'.join(lines),encoding='utf-8')
write_json(OUTPUT_DIR/'transcript_cleanup_audit.json',gemma_cleanup['audit'])
# Detailed group/box/dialogue exports retain original OCR; cleaned text is in transcript files above.
metadata=json.loads((OUTPUT_DIR/'run_metadata.json').read_text(encoding='utf-8'))
metadata['transcript_cleanup']=dict(model=GEMMA_IDENTITY,stage=stage_ref('gemma_transcript'),
    changed_entries=sum(row['changed'] for row in gemma_cleanup['audit']),raw_export='transcript_raw.jsonl')
metadata['stages']=manifest['stages']
write_json(OUTPUT_DIR/'run_metadata.json',metadata)
print('Gemma cleanup:',len(gemma_cleanup['audit']),'entries;',sum(r['changed'] for r in gemma_cleanup['audit']),'changed')


print('Sequence inference complete:', SEQUENCE_ID)
