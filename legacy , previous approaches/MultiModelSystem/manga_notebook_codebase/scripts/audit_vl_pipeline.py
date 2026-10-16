"""Verify saved VL data and real CPU collator outputs; never train or edit datasets."""
import ast
from collections import Counter
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import re
import random
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/diagnostics/vl_pipeline_audit'
sys.path.insert(0, str(ROOT / 'src'))
from vl_prompts import OUTPUT_FORMAT, build_speaker_prompt, ordered_groups, parse_speaker_output
from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(notebook, namespace, names):
    # Extract functions only: do not execute notebook loading/training cells.
    for cell in load(ROOT / 'notebooks' / notebook)['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), notebook, 'exec'), namespace)


report = {'errors': [], 'warnings': [], 'splits': {}, 'tensor_examples': []}


def check(ok, message):
    if not ok:
        report['errors'].append(message)


ns = {'unicodedata': unicodedata}
functions('transcript_prep_VL.ipynb', ns, {'normalize_text', 'edit_distance', 'cer', 'align_group'})
# Equivalent Levenshtein implementation makes the complete replay faster.
try:
    from rapidfuzz.distance import Levenshtein
    ns['edit_distance'] = Levenshtein.distance
except ImportError:
    pass
render_ns = {'Image': Image, 'ImageDraw': ImageDraw, 'ImageFont': ImageFont, 'Path': Path}
functions('transcript_prep_VL.ipynb', render_ns, {'draw_annotated_page'})
# Use the exact drawing code, returning pixels before its file-writing lines.
nb = load(ROOT / 'notebooks/transcript_prep_VL.ipynb')
source = next(''.join(c['source']) for c in nb['cells'] if 'def draw_annotated_page(' in ''.join(c['source']))
source = source.split('    output_path = Path(output_path)')[0] + '    return image\n'
exec(source, render_ns)
split = load(ROOT / 'outputs/split.json')
check(not set(split['train']) & set(split['val']), 'Training/validation sequence overlap')
labels = {r['sequence_id']: r for r in rows(ROOT / 'TaskData/development/labels.jsonl')}
manifest = {r['sequence_id']: r for r in load(ROOT / 'TaskData/sequences.json')}
all_examples = []

for name in ['train', 'val']:
    prep = ROOT / 'outputs/transcript_prep_vl' / name
    prompt_dir = ROOT / 'outputs/speaker_prompts' / name
    summary = load(prep / 'summary.json')
    prompt_summary = load(prompt_dir / 'summary.json')
    if prompt_summary.get('output_format') != OUTPUT_FORMAT:
        raise SystemExit('Rerun build_speaker_prompt_Vl.ipynb for train and val before auditing the new speaker-line pipeline.')
    candidates_path = ROOT / 'outputs/text_extraction' / name / 'candidates.jsonl'
    candidates = {r['sequence_id']: r for r in rows(candidates_path)}
    inputs = {r['sequence_id']: r for r in rows(prep / 'inputs.jsonl')}
    targets = {r['sequence_id']: r for r in rows(prep / 'targets.jsonl')}
    audits = {(r['sequence_id'], r['group_id']): r for r in rows(prep / 'alignment_audit.jsonl')}
    prompts = {r['sequence_id']: r for r in rows(prompt_dir / 'prompts.jsonl')}
    cache = load(prep / 'annotated_images/cache.json')
    check(set(inputs) == set(candidates) == set(targets) == set(prompts) == set(split[name]), f'{name}: sequence membership')
    for file, key in [('inputs.jsonl', 'inputs'), ('targets.jsonl', 'targets'), ('alignment_audit.jsonl', 'audit')]:
        check(digest(prep / file) == summary[key + '_sha256'], f'{name}: preparation hash {file}')
    for file, key in [('inputs.jsonl', 'inputs'), ('targets.jsonl', 'targets')]:
        check(digest(prep / file) == prompt_summary['source_sha256'][key], f'{name}: prompt source hash {file}')
    check(digest(candidates_path) == summary['candidates_sha256'], f'{name}: candidate source hash')
    check(digest(ROOT / 'TaskData/development/labels.jsonl') == summary['labels_sha256'], f'{name}: reference source hash')
    check(digest(ROOT / 'src/vl_prompts.py') == prompt_summary['shared_prompt_sha256'], f'{name}: prompt code hash')
    for file, saved_hash in prompt_summary['output_sha256'].items():
        check(digest(prompt_dir / file) == saved_hash, f'{name}: prompt output hash {file}')
    counts = Counter()
    target_counts = Counter()
    for sid, record in inputs.items():
        canonical = {}
        for page in labels[sid]['pages']:
            for ref in page:
                speaker = ref['speaker']
                if speaker not in {'UNKNOWN', 'NARRATION'} and speaker not in canonical:
                    canonical[speaker] = f'char{len(canonical)+1}'
        expected_targets = {}
        expected_images = []
        for pi, page in enumerate(record['pages']):
            check(page['page_index'] == pi, f'{sid}: page index')
            expected = [{'id': f"p{pi}_{g['id']}", 'text': g['text'], 'box': g['box']} for g in candidates[sid]['pages'][pi]]
            check(page['groups'] == expected, f'{sid}/{pi}: OCR text/box/ID mismatch')
            original = ROOT / 'TaskData' / manifest[sid]['images'][pi]
            annotated = prep / page['image']
            expected_images.append(str(annotated.resolve()))
            entry = cache[page['image']]
            check(entry['fingerprint']['image_sha256'] == digest(original), f'{sid}/{pi}: original image cache hash')
            check(entry['fingerprint']['groups'] == [{'id': g['id'], 'box': g['box']} for g in page['groups']], f'{sid}/{pi}: cache boxes')
            with Image.open(annotated) as saved:
                check(list(saved.size) == page['image_size'], f'{sid}/{pi}: image dimensions')
                drawn = render_ns['draw_annotated_page'](original, page['groups'], None, summary['settings']['font_size'], summary['settings']['line_width'])
                check(drawn.tobytes() == saved.convert('RGB').tobytes(), f'{sid}/{pi}: cached annotation pixels differ')
                drawn.close()
            relative = str(annotated.relative_to(prep))
            check(digest(annotated) == prompt_summary['annotated_image_sha256'][relative], f'{sid}/{pi}: prompt image hash')
            for group in page['groups']:
                audit = audits[(sid, group['id'])]
                settings = summary['settings']
                result = ns['align_group'](group['text'], labels[sid]['pages'][pi], settings['cer_threshold'], settings['ambiguity_margin'], settings['min_fragment_chars'], settings['min_fragment_coverage'])
                for key, value in result.items():
                    check(audit.get(key) == value, f'{sid}/{group["id"]}: alignment replay {key}')
                counts[audit['status']] += 1
                if result['status'] == 'accepted':
                    speaker = canonical.get(result['speaker'], result['speaker'])
                    expected_targets[group['id']] = speaker
                    target_counts[speaker] += 1
                    check(audit['assigned_speaker'] == speaker, f'{sid}/{group["id"]}: canonical speaker')
        actual = {g['group_id']: g['speaker'] for p in targets[sid]['pages'] for g in p}
        check(actual == expected_targets, f'{sid}: targets differ from replayed references')
        check(prompts[sid]['prompt'] == build_speaker_prompt(ordered_groups(record['pages'])), f'{sid}: full inference prompt')
        check(prompts[sid]['images'] == expected_images, f'{sid}: inference image order')
    if name == 'train':
        all_examples = rows(prompt_dir / 'training_examples.jsonl')
        for example in all_examples:
            sid = example['sequence_id']
            answer = {g['group_id']: g['speaker'] for p in targets[sid]['pages'] for g in p}
            check(parse_speaker_output(example['assistant_response'], example['requested_groups']) == answer, f'{sid}: assistant answer')
            requested_pages = [dict(p, groups=[g for g in p['groups'] if g['id'] in answer]) for p in inputs[sid]['pages']]
            check(example['prompt'] == build_speaker_prompt(ordered_groups(requested_pages)), f'{sid}: supervised-only training prompt')
            check(example['images'] == prompts[sid]['images'], f'{sid}: training images order')
    report['splits'][name] = dict(sequences=len(inputs), groups=sum(counts.values()), alignment=dict(counts), speakers=dict(target_counts))
    print('Verified data, labels and annotation pixels:', name, flush=True)

training_dir = ROOT / 'outputs/vl_training/ordered_speaker_lora_v1'
metadata_path = training_dir / 'training_metadata.json'
if metadata_path.exists():
    metadata = load(ROOT / 'outputs/vl_training/ordered_speaker_lora_v1/training_metadata.json')
    check(metadata['training_examples_sha256'] == digest(ROOT / 'outputs/speaker_prompts/train/training_examples.jsonl'), 'Checkpoint training examples differ from current examples')
    check(metadata['split_sha256'] == digest(ROOT / 'outputs/split.json'), 'Checkpoint split differs')
    for file, saved_hash in metadata['image_sha256'].items():
        check(digest(ROOT / 'outputs/transcript_prep_vl/train' / file) == saved_hash, f'Checkpoint image differs: {file}')
    report['adapter_module_count'] = len(metadata['trainable_modules'])
    check(all('.language_model.layers.' in m for m in metadata['trainable_modules']), 'Adapter includes non-decoder modules')
else:
    metadata = dict(model_path=r'C:\MODELS\HF\Qwen2.5-VL-3B-Instruct',
                    min_pixels=256*28*28, max_pixels=768*28*28, max_sequence_tokens=6144)
    report['checkpoint_status'] = 'New-format training has not run yet.'

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vl_prompts import permute_speaker_labels
processor = AutoProcessor.from_pretrained(metadata['model_path'], local_files_only=True)
tensor_ns = dict(Image=Image, process_vision_info=process_vision_info, torch=torch, processor=processor,
                 parse_speaker_output=parse_speaker_output, permute_speaker_labels=permute_speaker_labels,
                 MIN_PIXELS=metadata['min_pixels'], MAX_PIXELS=metadata['max_pixels'], MAX_SEQUENCE_TOKENS=metadata['max_sequence_tokens'])
functions('train_VL.ipynb', tensor_ns, {'collate_example'})
random.seed(42)  # Reproduce the audit's presentation permutations.
total_tokens = speaker_tokens = 0
for i, example in enumerate(all_examples):
    batch = tensor_ns['collate_example']([example])
    ids, target = batch['input_ids'][0], batch['labels'][0]
    positions = torch.where(target != -100)[0]
    decoded = processor.tokenizer.decode(target[positions], skip_special_tokens=True).strip()
    original = list(parse_speaker_output(example['assistant_response'], example['requested_groups']).values())
    shuffled = list(parse_speaker_output(decoded, example['requested_groups']).values())
    check(all((a == b) == (c == d) for a, c in zip(original, shuffled) for b, d in zip(original, shuffled)), f'{example["sequence_id"]}: permutation changes identity partitions')
    check(all(a == b for a, b in zip(original, shuffled) if a in {'UNKNOWN', 'NARRATION'}), f'{example["sequence_id"]}: special class changed')
    check(torch.equal(target[positions], ids[positions]), f'{example["sequence_id"]}: labels differ from input tokens')
    check((target[:positions[0]] == -100).all().item(), f'{example["sequence_id"]}: prompt tokens contribute to loss')
    check(batch['image_grid_thw'].shape[0] == 3 and batch['pixel_values'].numel() > 0, f'{example["sequence_id"]}: three images missing from tensors')
    answer = decoded
    encoding = processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    spans = [m.span() for m in re.finditer(r'char[1-9][0-9]*|UNKNOWN|NARRATION', answer)]
    value_tokens = sum(any(a < end and b > start for start, end in spans) for a, b in encoding['offset_mapping'])
    total_tokens += len(positions)
    speaker_tokens += value_tokens
    report['tensor_examples'].append(dict(sequence_id=example['sequence_id'], tokens=len(ids), supervised_tokens=len(positions), speaker_value_tokens=value_tokens, grids=batch['image_grid_thw'].tolist()))
    del batch
    if (i+1) % 10 == 0:
        print('Verified CPU tensors:', i+1, '/', len(all_examples), flush=True)

from transformers.loss.loss_utils import ForCausalLMLoss
report['installed_causal_loss_source'] = inspect.getsource(ForCausalLMLoss)
# Check the installed loss against explicit next-token cross entropy.
logits = torch.randn(1, 5, 7, generator=torch.Generator().manual_seed(42))
test_labels = torch.tensor([[-100, -100, 2, 4, 1]])
actual_loss = ForCausalLMLoss(logits, test_labels, vocab_size=7)
expected_loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7), test_labels[:, 1:].reshape(-1), ignore_index=-100)
check(torch.allclose(actual_loss, expected_loss), 'Installed causal loss shift/mask incorrect')
report['loss'] = dict(supervised_tokens=total_tokens, speaker_value_tokens=speaker_tokens, speaker_value_fraction=speaker_tokens/total_tokens, objective='Next-token cross entropy over permuted speaker lines and chat ending; prompt and images masked')

# Inspect saved adapter tensors without loading the base model or using the GPU.
from safetensors import safe_open
checkpoints = sorted(training_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
if checkpoints:
    checkpoint = checkpoints[-1]
    with safe_open(str(checkpoint / 'adapter_model.safetensors'), framework='pt', device='cpu') as adapter:
        keys = list(adapter.keys())
        for key in keys:
            check('.language_model.layers.' in key and 'lora_' in key, f'Unexpected adapter tensor: {key}')
            tensor = adapter.get_tensor(key)
            check(torch.isfinite(tensor).all().item(), f'Nonfinite adapter tensor: {key}')
            if 'lora_B' in key:
                check(tensor.count_nonzero().item() > 0, f'Unchanged adapter B tensor: {key}')
    report['checkpoint'] = dict(tensors=len(keys), epoch=load(checkpoint / 'trainer_state.json')['epoch'])

# Recompute saved validation scores with the notebook's exact scoring function.
spec = importlib.util.spec_from_file_location('scorer', ROOT / 'TaskData/score.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
score_ns = {'scorer': scorer}
functions('validation_VL.ipynb', score_ns, {'score_sequence'})
gold = {r['sequence_id']: r['speakers'] for r in rows(ROOT / 'outputs/speaker_prompts/val/scoring_targets.jsonl')}
run = ROOT / 'outputs/vl_inference/val/qwen25vl3b_speaker_lines_baseline_v1'
if (run / 'evaluation.jsonl').exists():
    predictions = rows(run / 'predictions.jsonl')
    evaluations = {r['sequence_id']: r for r in rows(run / 'evaluation.jsonl')}
    correct = majority_correct = 0
    for prediction in predictions:
        sid = prediction['sequence_id']
        result = score_ns['score_sequence'](prediction['speakers'], gold[sid])
        for key, value in result.items():
            check(evaluations[sid][key] == value, f'{sid}: saved scoring differs: {key}')
        correct += result['correct']
        majority_correct += max(Counter(s for s in gold[sid].values() if s not in {'UNKNOWN', 'NARRATION'}).values(), default=0)
    report['evaluation'] = dict(correct=correct, labelled_groups=sum(map(len, gold.values())), single_character_baseline_correct=majority_correct,
        predicted_speakers=dict(Counter(s for p in predictions for s in p['speakers'].values())), single_label_sequences=sum(len(set(p['speakers'].values())) == 1 for p in predictions))
report['warnings'] = [
    'Character labels are shuffled per presentation. Exact-token loss is still not permutation invariant.',
    'Training requests accepted groups only; validation requests every group, including rejected alignment groups.',
    'CER alignment verifies text agreement, not visual ground truth of the matched bubble.',
    'This audit does not train, generate model predictions, or establish visual reasoning accuracy.',
]
train_ocr = load(ROOT / 'outputs/text_extraction/train/candidates.metadata.json')
val_ocr = load(ROOT / 'outputs/text_extraction/val/candidates.metadata.json')
if train_ocr['code'] != val_ocr['code'] or train_ocr['sound_effects_sha256'] != val_ocr['sound_effects_sha256']:
    report['warnings'].append('Train/validation OCR filtering versions differ; rebuild both splits.')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'errors': report['errors'], 'splits': report['splits'], 'loss': report['loss'], 'report': str(OUT / 'report.json')}, indent=2))
raise SystemExit(1 if report['errors'] else 0)
