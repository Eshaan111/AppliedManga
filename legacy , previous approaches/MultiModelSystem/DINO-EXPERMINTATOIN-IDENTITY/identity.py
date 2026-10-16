"""Standalone manual-crop DINO experiment. No OCR or speaker linking."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path('C:/MODELS/dinov2-small')
STRATEGIES = ('face', 'head_hair', 'upper_body')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_annotations(path=ROOT / 'annotations/characters.json'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Expected annotation schema_version 1')
    seen, rows, sequence_ids = set(), [], set()
    task_root = Path(data['task_data_root'])
    if not task_root.is_absolute():
        task_root = ROOT / task_root
    for seq in data['sequences']:
        sid = seq['sequence_id']
        if sid in sequence_ids:
            raise ValueError(f'Duplicate sequence (calibration and heldout must be disjoint): {sid}')
        sequence_ids.add(sid)
        if seq['split'] not in ('calibration', 'heldout'):
            raise ValueError(f'Invalid split: {sid}')
        page_indices = set()
        for page in seq['pages']:
            pi = page['page_index']
            if not isinstance(pi, int) or pi < 0 or pi in page_indices:
                raise ValueError(f'Invalid or duplicate page_index: {sid} / {pi}')
            page_indices.add(pi)
            image_path = (task_root / page['image']).resolve()
            if not image_path.is_relative_to(task_root.resolve()):
                raise ValueError('Page path escapes TaskData')
            with Image.open(image_path) as image:
                width, height = image.size
            for char in page['characters']:
                cid = char['crop_id']
                if not cid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in sid + cid):
                    raise ValueError('Sequence and crop IDs must be filesystem-safe')
                key = (sid, cid)
                if key in seen:
                    raise ValueError(f'Duplicate crop: {key}')
                seen.add(key)
                label = char.get('identity_label') or None
                if label is not None and not isinstance(label, str):
                    raise ValueError('Identity labels must be strings or null')
                for strategy, box in char['boxes'].items():
                    if strategy not in STRATEGIES:
                        raise ValueError(f'Unknown strategy: {strategy}')
                    if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                            or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                        raise ValueError(f'Invalid {strategy} box for {key}: {box}')
                    pixel_box = list(map(int, box))
                    if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
                        raise ValueError('Box becomes empty after conversion to pixels')
                    rows.append(dict(sequence_id=sid, split=seq['split'], page_index=page['page_index'],
                                     crop_id=cid, strategy=strategy, identity_label=label,
                                     box=pixel_box, image_path=str(image_path)))
    return data, sorted(rows, key=lambda r: (r['sequence_id'], r['page_index'], r['crop_id'], r['strategy']))


def export_crops(rows, destination=ROOT / 'character_crops'):
    result = []
    for row in rows:
        output = Path(destination) / row['strategy'] / row['sequence_id'] / (row['crop_id'] + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row['image_path']) as image:
            image.convert('RGB').crop(row['box']).save(output)
        result.append(dict(row, crop_path=str(output)))
    return result


def load_model(model_path=MODEL_PATH, device=None):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True)
    return processor, model.to(device).eval(), device


def embed(rows, processor, model, device, batch_size=16):
    import torch
    import torch.nn.functional as F
    if not rows or batch_size < 1:
        raise ValueError('Add manual character crops first; batch_size must be positive')
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(row['crop_path']) as image:
                    images.append(image.convert('RGB'))
            inputs = processor(images=images, return_tensors='pt').to(device)
            output = model(**inputs)
            vectors.append(F.normalize(output.pooler_output.float(), dim=-1).cpu().numpy())
    result = np.concatenate(vectors)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
        raise ValueError('Invalid normalized embeddings')
    return result


def groups(rows):
    result = {}
    for i, row in enumerate(rows):
        result.setdefault((row['sequence_id'], row['strategy']), []).append(i)
    return result


def make_pairs(rows, vectors):
    pairs = []
    for (sid, strategy), indices in groups(rows).items():
        for offset, i in enumerate(indices):
            for j in indices[offset + 1:]:
                a, b = rows[i], rows[j]
                if a['page_index'] == b['page_index'] or not a['identity_label'] or not b['identity_label']:
                    continue
                pairs.append(dict(sequence_id=sid, strategy=strategy, split=a['split'],
                                  crop1=a['crop_id'], crop2=b['crop_id'],
                                  index1=i, index2=j, similarity=float(np.clip(vectors[i] @ vectors[j], -1, 1)),
                                  same_character=a['identity_label'] == b['identity_label']))
    return pairs


def metrics(pairs, threshold):
    if not pairs:
        return {'count': 0}
    y = np.array([p['same_character'] for p in pairs], dtype=bool)
    prediction = np.array([p['similarity'] >= threshold for p in pairs])
    tp, tn = int(sum(prediction & y)), int(sum(~prediction & ~y))
    fp, fn = int(sum(prediction & ~y)), int(sum(~prediction & y))
    same = np.array([p['similarity'] for p in pairs if p['same_character']])
    different = np.array([p['similarity'] for p in pairs if not p['same_character']])
    from sklearn.metrics import roc_auc_score
    return dict(count=len(pairs), same_count=len(same), different_count=len(different),
                same_mean=float(same.mean()) if len(same) else None,
                same_std=float(same.std()) if len(same) else None,
                different_mean=float(different.mean()) if len(different) else None,
                different_std=float(different.std()) if len(different) else None,
                accuracy=(tp + tn) / len(pairs),
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2 if len(same) and len(different) else None,
                precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                roc_auc=float(roc_auc_score(y, [p['similarity'] for p in pairs])) if len(same) and len(different) else None,
                true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)


def calibrate(pairs):
    if not pairs or any(p['split'] != 'calibration' for p in pairs):
        raise ValueError('Threshold tuning requires calibration pairs only')
    if {p['same_character'] for p in pairs} != {True, False}:
        raise ValueError('Need both same and different cross-page pairs for calibration')
    scores = sorted(set(p['similarity'] for p in pairs))
    thresholds = [float(np.nextafter(scores[0], -np.inf))]
    thresholds += [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    thresholds += [float(np.nextafter(scores[-1], np.inf))]
    # Balanced accuracy avoids a mostly-different dataset favoring "all different".
    candidates = [(t, metrics(pairs, t)) for t in thresholds]
    threshold, report = max(candidates, key=lambda x: (x[1]['balanced_accuracy'], x[1]['f1'], x[0]))
    return dict(threshold=threshold, objective='balanced_accuracy', calibration=report)


def cluster(rows, vectors, threshold):
    if not math.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    assignments = {}
    for _, indices in groups(rows).items():
        clusters = []
        for i in indices:
            best, similarity = None, -float('inf')
            for k, entry in enumerate(clusters):
                score = float(vectors[i] @ entry['centroid'])
                if score > similarity:
                    best, similarity = k, score
            if best is None or similarity < threshold:
                clusters.append({'members': [i], 'centroid': vectors[i].copy()})
                best = len(clusters) - 1
            else:
                clusters[best]['members'].append(i)
                mean = vectors[clusters[best]['members']].mean(axis=0)
                norm = np.linalg.norm(mean)
                clusters[best]['centroid'] = mean / norm if norm > 1e-12 else vectors[i].copy()
            assignments[i] = f'char{best + 1}'
    return assignments


def plot_matrix(rows, vectors, indices, output=None):
    import matplotlib.pyplot as plt
    ids = [rows[i]['crop_id'] for i in indices]
    fig, ax = plt.subplots(figsize=(max(6, len(ids) * .4), max(5, len(ids) * .4)))
    matrix = np.clip(vectors[indices] @ vectors[indices].T, -1, 1)
    view = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(ids)), ids, rotation=90)
    ax.set_yticks(range(len(ids)), ids)
    ax.set_title(f"{rows[indices[0]]['sequence_id']} / {rows[indices[0]]['strategy']}")
    fig.colorbar(view, ax=ax, label='Cosine similarity')
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=140)
        plt.close(fig)
    return fig


def plot_distributions(pairs, threshold, output=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for same, label in ((True, 'same'), (False, 'different')):
        ax.hist([p['similarity'] for p in pairs if p['same_character'] == same],
                bins=np.linspace(-1, 1, 41), alpha=.55, label=label)
    ax.axvline(threshold, color='black', linestyle='--', label=f'threshold {threshold:.3f}')
    ax.set(xlabel='Cross-page cosine similarity', ylabel='Pairs')
    ax.legend()
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=140)
        plt.close(fig)
    return fig


def plot_failures(rows, pairs, threshold, output, limit=12):
    import matplotlib.pyplot as plt
    errors = [p for p in pairs if (p['similarity'] >= threshold) != p['same_character']]
    errors.sort(key=lambda p: abs(p['similarity'] - threshold), reverse=True)
    if not errors:
        return []
    selected = errors[:limit]
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, len(selected) * 2.5), squeeze=False)
    for n, pair in enumerate(selected):
        kind = 'false negative' if pair['same_character'] else 'false positive'
        for col, key in enumerate(('index1', 'index2')):
            row = rows[pair[key]]
            with Image.open(row['crop_path']) as image:
                axes[n, col].imshow(image.convert('RGB'))
            axes[n, col].set_title(f"{row['sequence_id']} / {row['crop_id']} ({row['identity_label']})\n{kind}: {pair['similarity']:.3f}", fontsize=8)
            axes[n, col].axis('off')
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return selected


def run_experiment(annotation_path, output, model_path=MODEL_PATH, batch_size=16):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    data, rows = load_annotations(annotation_path)
    if not rows:
        raise ValueError('No manual crops yet. Open annotations/annotate.html, draw boxes, and export characters.json.')
    rows = export_crops(rows)
    processor, model, device = load_model(model_path)
    vectors = embed(rows, processor, model, device, batch_size)
    pairs = make_pairs(rows, vectors)
    reports = {}
    assignments = {}
    for strategy in STRATEGIES:
        subset = [p for p in pairs if p['strategy'] == strategy]
        calibration = [p for p in subset if p['split'] == 'calibration']
        if {p['same_character'] for p in calibration} != {True, False}:
            reports[strategy] = {'status': 'insufficient labelled calibration pairs'}
            continue
        report = calibrate(calibration)
        threshold = report['threshold']
        report['heldout'] = metrics([p for p in subset if p['split'] == 'heldout'], threshold)
        reports[strategy] = report
        assignments.update({i: label for i, label in cluster(rows, vectors, threshold).items() if rows[i]['strategy'] == strategy})
    if not assignments:
        raise ValueError('Annotate cross-page SAME and DIFFERENT calibration examples for at least one strategy')
    output.mkdir(parents=True)
    np.save(output / 'embeddings.npy', vectors)
    save_json(output / 'crop_index.json', rows)
    save_json(output / 'pairs.json', pairs)
    save_json(output / 'metrics.json', reports)
    save_json(output / 'character_identity_labels.json', {
        seq['sequence_id']: {c['crop_id']: c.get('identity_label') for p in seq['pages'] for c in p['characters']}
        for seq in data['sequences']})
    save_json(output / 'run_config.json', dict(model_path=str(Path(model_path).resolve()), device=device,
              annotation_path=str(Path(annotation_path).resolve()), annotation_snapshot=data,
              pooling='pooler_output (CLS)', normalization='L2', pair_scope='within sequence, cross-page, same strategy',
              clustering='greedy normalized mean centroid; ordered by page_index and crop_id'))
    for (sid, strategy), indices in groups(rows).items():
        matrix = np.clip(vectors[indices] @ vectors[indices].T, -1, 1)
        np.save(output / f'{sid}_{strategy}_similarity.npy', matrix)
        plot_matrix(rows, vectors, indices, output / f'{sid}_{strategy}_similarity.png')
    for strategy, report in reports.items():
        if 'threshold' not in report:
            continue
        for split in ('calibration', 'heldout'):
            subset = [p for p in pairs if p['strategy'] == strategy and p['split'] == split]
            if subset:
                plot_distributions(subset, report['threshold'], output / f'{strategy}_{split}_distributions.png')
                failures = plot_failures(rows, subset, report['threshold'], output / f'{strategy}_{split}_failures.png')
                save_json(output / f'{strategy}_{split}_failures.json', failures)
    with (output / 'character_results.jsonl').open('w', encoding='utf-8') as stream:
        for seq in data['sequences']:
            for strategy in STRATEGIES:
                if 'threshold' not in reports[strategy]:
                    continue
                for page in seq['pages']:
                    characters = [dict(id=r['crop_id'], box=r['box'], identity=assignments[i]) for i, r in enumerate(rows)
                                  if r['sequence_id'] == seq['sequence_id'] and r['strategy'] == strategy and r['page_index'] == page['page_index']]
                    stream.write(json.dumps(dict(sequence_id=seq['sequence_id'], page_index=page['page_index'],
                                                 crop_strategy=strategy, characters=characters)) + '\n')
    return reports
