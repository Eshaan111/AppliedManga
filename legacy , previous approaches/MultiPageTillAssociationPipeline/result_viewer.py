"""Read saved page results for inspection. Never run detection, OCR or VL models."""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import pipeline_helpers as P


def sample_path(item):
    return P.ROOT / 'data/training/joint_targets' / item['split'] / item['sequence_id'] / f"page_{item['page_number']:02d}/sample.json"


def result_path(item, stage, run_folder=None):
    names = {'bubbles':'bubbles.json', 'ocr':'ocr.json',
             'cleanup':'bubble_gt_labels.json', 'annotation':'annotation.json'}
    if stage == 'split':
        return Path(item['image_path'])
    if stage in names:
        return P.page_folder(item) / names[stage]
    if stage in ['target', 'training']:
        return sample_path(item)
    if stage == 'validation':
        return Path(run_folder) / 'validation.json'
    raise ValueError(f'Unknown stage: {stage}')


def validation_row(item, run_folder):
    path = result_path(item, 'validation', run_folder)
    if not path.is_file():
        return None
    for row in P.read_json(path)['results']:
        if row['sequence_id'] == item['sequence_id'] and row['page_number'] == item['page_number'] and row['split'] == item['split']:
            return row
    return None


def has_result(item, stage, run_folder=None):
    if stage == 'validation':
        return validation_row(item, run_folder) is not None
    return result_path(item, stage, run_folder).is_file()


def show_image(path, boxes=(), labels=None):
    with Image.open(path) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    labels = labels or {}
    for record in boxes:
        box = record['box']
        bubble_id = record['bubble_id']
        label = labels.get(bubble_id, bubble_id)
        draw.rectangle(box, outline='blue', width=3)
        x, y = box[:2]
        draw.text((x + 3, y + 3), label, font=font, fill='blue', stroke_width=2, stroke_fill='white')
    # Resize only the displayed copy. Saved images are untouched.
    image.thumbnail((1000, 1400))
    display(image)


def show_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def show_page(item, stage, run_folder=None):
    print(f"\n{item['split']} | {item['sequence_id']} | page {item['page_number']}")
    if not has_result(item, stage, run_folder):
        print('No saved result yet for this stage.')
        return
    if stage == 'split':
        show_image(item['image_path'])
        show_json(item)
        return
    if stage in ['bubbles', 'ocr', 'cleanup']:
        bubble_file = 'bubbles_clean.json' if stage == 'cleanup' else 'bubbles.json'
        bubbles = P.load_stage(item, bubble_file)
        labels = {}
        if stage == 'cleanup':
            gt = P.load_stage(item, 'bubble_gt_labels.json')
            labels = {b['bubble_id']:f"{b['bubble_id']} : {b['speaker'] or 'unmatched'}" for b in gt['bubbles']}
        show_image(item['image_path'], bubbles, labels)
        if stage == 'bubbles':
            show_json(bubbles)
        elif stage == 'ocr':
            show_json(P.load_stage(item, 'ocr.json'))
        else:
            show_json(gt['bubbles'])
        return
    if stage == 'annotation':
        annotation = P.load_stage(item, 'annotation.json')
        path = P.page_folder(item) / annotation['image']
        if path.is_file():
            show_image(path)
        else:
            print('The saved annotated image is missing.')
        return
    path = sample_path(item)
    if not path.is_file():
        print('The exported sample is missing.')
        return
    sample = P.read_json(path)
    show_image(path.parent / sample['image'])
    print('\nInput prompt:')
    print(sample['conversations'][0]['value'])
    print('\nAssistant target:')
    show_json(json.loads(sample['conversations'][1]['value']))
    if stage == 'target':
        print('\nOriginal -> training IDs:')
        show_json(sample['id_maps'])
    elif stage == 'training':
        run_folder = Path(run_folder)
        paths_file = run_folder / 'sample_paths.json'
        if paths_file.is_file():
            used = {str(Path(p).resolve()) for p in P.read_json(paths_file)}
            print('\nIncluded in this training run:', str(path.resolve()) in used)
        history = run_folder / 'loss_history.json'
        if history.is_file():
            print('\nSaved epoch losses (run-wide, not per page):')
            show_json(P.read_json(history))
        else:
            print('\nNo saved training losses for this run yet.')
        print('Training saves targets and run losses; generated page predictions are saved by the validation notebook.')
    elif stage == 'validation':
        row = validation_row(item, run_folder)
        print('\nGenerated response:')
        print(row['response'])
        print('\nSaved metrics:')
        show_json({k:v for k,v in row.items() if k not in ['response','prediction','target']})


def show_random(stage, split='train', run_folder=None):
    available = [item for item in P.pages((split,)) if has_result(item, stage, run_folder)]
    if not available:
        print(f'No saved {stage} results for {split} yet.')
        return None
    item = random.choice(available)
    show_page(item, stage, run_folder)
    return item


def show_sequence(sequence_id, stage, split='train', run_folder=None):
    items = [item for item in P.pages((split,)) if item['sequence_id'] == sequence_id]
    if not items:
        print('Sequence not found in this split. Choose the correct sequence and split.')
        return
    for item in sorted(items, key=lambda item:item['page_number']):
        show_page(item, stage, run_folder)
