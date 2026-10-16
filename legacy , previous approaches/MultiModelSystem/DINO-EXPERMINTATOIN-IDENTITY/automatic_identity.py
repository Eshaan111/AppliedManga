"""Face-only DINO embeddings and cross-page cosine pairs for human review."""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import identity as dino


def load_crops(manifest_path, max_pages=3):
    if max_pages < 1:
        raise ValueError('Page limit must be positive')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Expected detection schema_version 1')
    selected = {}
    for page in sorted(manifest['pages'], key=lambda p: (p['sequence_id'], p['page_index'])):
        indices = selected.setdefault(page['sequence_id'], [])
        if len(indices) < max_pages:
            indices.append(page['page_index'])
    rows = [dict(row, identity_label=None) for row in manifest['rows']
            if row['page_index'] in selected[row['sequence_id']] and row['strategy'] == 'face']
    manifest['pages'] = [page for page in manifest['pages']
                         if page['page_index'] in selected[page['sequence_id']]]
    return manifest, sorted(rows, key=lambda r: (r['sequence_id'], r['page_index'], r['crop_id'], r['strategy']))


def write_review_html(pairs, output, threshold):
    """Static side-by-side crops; all pairs remain available in the CSV."""
    import html
    cards = []
    for pair in pairs[:200]:
        images = ''.join(
            f'<figure><img src="{html.escape(Path(pair[f"crop_path{k}"]).as_uri(), quote=True)}">'
            f'<figcaption>page {pair[f"page{k}"]}: {html.escape(pair[f"crop{k}"])}</figcaption></figure>'
            for k in (1, 2))
        cards.append(f'<article><h3>{html.escape(pair["sequence_id"])} | cosine {pair["similarity"]:.4f}'
                     f' | {pair["prediction"]}</h3><div>{images}</div></article>')
    Path(output).write_text('<!doctype html><meta charset="utf-8"><title>DINO pair review</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#eee}article{background:white;padding:16px;margin:16px 0}'
        'article div{display:flex;gap:20px}figure{margin:0}img{width:220px;height:220px;object-fit:contain}</style>'
        f'<h1>DINO face pair review</h1><p>{len(pairs)} cross-page pairs; showing up to 200 in descending score order per sequence. '
        f'Threshold {threshold} is exploratory. Record same/different in pair_review.csv. '
        'High similarity is a hypothesis, not a verified identity.</p>' + ''.join(cards), encoding='utf-8')


def run_identity(manifest_path, output, model_path=dino.MODEL_PATH, threshold=.8,
                 max_pages=3, batch_size=16, labels_path=None):
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Cosine threshold must be finite and in [-1, 1]')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    manifest, rows = load_crops(manifest_path, max_pages)
    if not rows:
        raise ValueError('No detected crops. Inspect overlays or lower detector confidence first.')
    processor, model, device = dino.load_model(model_path)
    vectors = dino.embed(rows, processor, model, device, batch_size)
    output.mkdir(parents=True)
    np.save(output / 'embeddings.npy', vectors)
    dino.save_json(output / 'crop_index.json', rows)
    dino.save_json(output / 'run_config.json', dict(model_path=str(model_path), device=device,
                   threshold=threshold, threshold_status='Exploratory, not calibrated',
                   max_pages_per_sequence=max_pages, detection_manifest=str(Path(manifest_path).resolve()),
                   detector_config=manifest['config'], pooling='CLS pooler_output', normalization='L2',
                   model_id='facebook/dinov2-small', clustering=False))
    pairs = []
    for (sid, strategy), indices in dino.groups(rows).items():
        matrix = np.clip(vectors[indices] @ vectors[indices].T, -1, 1)
        np.save(output / f'{sid}_{strategy}_similarity.npy', matrix)
        dino.plot_matrix(rows, vectors, indices, output / f'{sid}_{strategy}_similarity.png')
        for offset, i in enumerate(indices):
            for j in indices[offset + 1:]:
                if rows[i]['page_index'] != rows[j]['page_index']:
                    pairs.append(dict(sequence_id=sid, strategy=strategy, crop1=rows[i]['crop_id'],
                                      crop2=rows[j]['crop_id'], page1=rows[i]['page_index'], page2=rows[j]['page_index'],
                                      crop_path1=rows[i]['crop_path'], crop_path2=rows[j]['crop_path'],
                                      similarity=float(np.clip(vectors[i] @ vectors[j], -1, 1)),
                                      prediction='probably same' if vectors[i] @ vectors[j] >= threshold else 'probably different',
                                      same_character='', notes=''))
    dino.save_json(output / 'cross_page_pairs.json', pairs)
    # Stable review IDs and explicit page/crop paths, with blank human labels.
    pairs.sort(key=lambda p: (p['sequence_id'], -p['similarity'], p['crop1'], p['crop2']))
    if pairs:
        with (output / 'pair_review.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    else:
        (output / 'pair_review.csv').write_text('sequence_id,crop1,crop2,similarity,same_character,notes\n', encoding='utf-8')
    write_review_html(pairs, output / 'pair_review.html', threshold)
    # Human labels are loaded only AFTER predictions have been fixed.
    if labels_path:
        labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
        evaluated = [dict(r, identity_label=labels.get(r['sequence_id'], {}).get(r['crop_id'])) for r in rows]
        if any(r['identity_label'] is not None and not isinstance(r['identity_label'], str) for r in evaluated):
            raise ValueError('Evaluation labels must be strings or null')
        labelled_pairs = dino.make_pairs(evaluated, vectors)
        dino.save_json(output / 'evaluation.json', {s: dino.metrics([p for p in labelled_pairs if p['strategy'] == s], threshold)
                                                  for s in sorted({r['strategy'] for r in rows})})
    summary = dict(crops=len(rows), cross_page_pairs=len(pairs),
                   quality='Unmeasured: manually label same/different pairs to assess DINO',
                   clustering=False)
    dino.save_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=dino.MODEL_PATH)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--pages', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or dino.ROOT / 'outputs' / f'identity_{datetime.now():%Y%m%d_%H%M%S_%f}'
    print(run_identity(args.detections, output, args.model, args.threshold, args.pages, args.batch_size, args.labels))
    print(f'Saved: {output}')
