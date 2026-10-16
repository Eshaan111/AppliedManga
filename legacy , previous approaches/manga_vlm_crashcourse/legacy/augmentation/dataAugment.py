import json
import itertools
from dataUtils import MangaSequenceDataset
from pathlib import Path

json_path = r'C:\Users\ESHAAN\BYTE\Applied-Task\manga_vlm_crashcourse\TaskData\development\labels.jsonl'
img_path = r'C:\Users\ESHAAN\BYTE\Applied-Task\manga_vlm_crashcourse\TaskData\development\images'
out_path = r'C:\Users\ESHAAN\BYTE\Applied-Task\manga_vlm_crashcourse\TaskData\development\augmented_labels.jsonl'

SPECIAL = {"NARRATION", "UNKNOWN"}


def relabel_characters(pages):
    mapping = {}
    out = []
    for page in pages:
        new_page = []
        for obj in page:
            sp = obj['speaker']
            if sp not in SPECIAL:
                if sp not in mapping:
                    mapping[sp] = f'character{len(mapping) + 1}'
                sp = mapping[sp]
            new_page.append({'speaker': sp, 'text': obj['text']})
        out.append(new_page)
    return out


def augment(json_path, img_path, perm_count=None, include_identity=True):
    dataset = MangaSequenceDataset(json_path, img_path)
    img_root = Path(img_path)
    results = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seq_id = sample["sequence_id"]
        image_paths = sorted(Path(p) for p in sample["image_paths"])
        target = sample["target"]
        assert len(image_paths) == len(target), f"{seq_id}: page/image mismatch"

        perms = itertools.islice(itertools.permutations(range(len(image_paths))), perm_count)
        for p_idx, perm in enumerate(perms):
            if p_idx == 0 and not include_identity:
                continue
            results.append({
                'sequence_id': f'{seq_id}PERM{p_idx}',
                'image_paths': [image_paths[j].relative_to(img_root).as_posix() for j in perm],
                'target': relabel_characters([target[j] for j in perm]),
            })
    return results


def save_jsonl(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == '__main__':
    data = augment(json_path, img_path, perm_count=None)   # 3 new perms per sequence
    save_jsonl(data, out_path)
    print(f'Wrote {len(data)} samples to {out_path}')

    loaded = load_jsonl(out_path)
    print(loaded[0]['sequence_id'], loaded[0]['image_paths'])