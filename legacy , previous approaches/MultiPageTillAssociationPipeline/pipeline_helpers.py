"""Plain file helpers. Importing this module does not load models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data/splits/manifest.json'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def development_dir():
    config = read_json(ROOT / 'project.json')
    local = ROOT / config['development_dir']
    return local if (local / 'labels.jsonl').is_file() else Path(config['fallback_development_dir'])

def split_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError('Run 00_split_data.ipynb first.')
    manifest = read_json(MANIFEST)
    train = set(manifest['sequences']['train'])
    val = set(manifest['sequences']['val'])
    if train & val:
        raise ValueError('A sequence appears in both splits.')
    seen = set()
    for item in manifest['pages']:
        key = (item['sequence_id'], item['page_number'])
        if key in seen or item['sequence_id'] not in set(manifest['sequences'][item['split']]):
            raise ValueError('Duplicate page or incorrect split membership.')
        seen.add(key)
    return manifest

def pages(splits=('train', 'val')):
    return [p for p in split_manifest()['pages'] if p['split'] in splits]

def page_folder(item):
    return ROOT / 'data/intermediate' / item['split'] / item['sequence_id'] / f"page_{item['page_number']:02d}"

def response_path(item):
    return ROOT / 'manual_labels/responses' / item['split'] / item['sequence_id'] / f"page_{item['page_number']:02d}.json"

def page_info(item):
    from PIL import Image
    path = Path(item['image_path'])
    if file_hash(path) != item['image_sha256']:
        raise ValueError(f'Source image changed: {path}. Keep the original split dataset.')
    with Image.open(path) as image:
        width, height = image.size
    metadata = {key:item[key] for key in ['sequence_id','page_number','split','image_path','image_sha256']}
    return dict(metadata, name=path.name, sha256=item['image_sha256'], width=width, height=height)

def cache_key(item, settings, dependencies):
    return {'image_sha256':item['image_sha256'], 'settings':settings,
            'dependencies':{str(Path(p).resolve()):file_hash(p) for p in dependencies}}

def cached(item, name, settings, dependencies=(), force=False):
    path = page_folder(item) / name
    if force or not path.is_file():
        return False
    saved = read_json(path)
    if name == 'annotation.json' and not (path.parent / saved['records']['image']).is_file():
        return False
    return saved['page'] == page_info(item) and saved.get('cache') == cache_key(item, settings, dependencies)

def save_stage(item, name, records, settings, dependencies=()):
    write_json(page_folder(item) / name, {'page':page_info(item),
        'cache':cache_key(item, settings, dependencies), 'records':records})

def load_stage(item, name):
    saved = read_json(page_folder(item) / name)
    if saved['page'] != page_info(item):
        raise ValueError(f'{name} belongs to another image or split.')
    return saved['records']

def stage_fingerprint(item):
    names = ['bubbles_clean.json','ocr_clean.json','bubble_gt_labels.json','characters.json']
    return {name:file_hash(page_folder(item) / name) for name in names}

def gt_sequence(item):
    manifest = split_manifest()
    path = Path(manifest['labels_path'])
    if file_hash(path) != manifest['labels_sha256']:
        raise ValueError('GT labels changed since the split was created.')
    with path.open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                entry = json.loads(line)
                if entry['sequence_id'] == item['sequence_id']:
                    return entry
    raise ValueError('Sequence has no ground truth.')

def exported_samples(split):
    # Use only exports belonging to the fixed manifest, never a broad glob.
    manifest = split_manifest()
    manifest_hash = file_hash(MANIFEST)
    result = []
    for item in pages((split,)):
        path = ROOT / 'data/training/joint_targets' / split / item['sequence_id'] / f"page_{item['page_number']:02d}/sample.json"
        if not path.is_file():
            continue
        sample = read_json(path)
        if sample['split'] != split or sample['sequence_id'] != item['sequence_id'] or sample['page_number'] != item['page_number']:
            raise ValueError(f'Export metadata differs: {path}')
        if sample['split_manifest_sha256'] != manifest_hash:
            raise ValueError('Export uses a different split manifest.')
        image = path.parent / sample['image']
        if not image.is_file() or file_hash(image) != sample['annotated_image_sha256']:
            raise ValueError(f'Exported image changed; rerun target export: {path}')
        dependencies = [response_path(item)] + [page_folder(item) / n for n in stage_fingerprint(item)]
        if sample['export_cache'] != cache_key(item, {'version':1}, dependencies):
            raise ValueError(f'Labels changed; rerun target export: {path}')
        result.append(path)
    if not result:
        raise ValueError(f'No reviewed {split} samples exported yet.')
    return result
