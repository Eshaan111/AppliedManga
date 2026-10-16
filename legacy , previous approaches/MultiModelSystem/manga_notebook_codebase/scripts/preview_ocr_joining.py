"""Generate OCR grouping previews and target comparison without model inference."""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data_utils import load_jsonl, save_jsonl
from dataset_utils import read_dataset, image_paths, save_json
from ocr_joining import join_ocr_candidates, compare_groups_to_targets, draw_ocr_groups
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=int, default=4)
    parser.add_argument('--page', type=int, choices=range(3), default=0)
    args = parser.parse_args()
    rows = read_dataset(ROOT, 'val')
    candidates = {r['sequence_id']: r['pages'] for r in load_jsonl(ROOT / 'outputs/ocr/paddleocr/candidates.jsonl')}
    refs = {r['sequence_id']: r['pages'] for r in load_jsonl(ROOT / 'datasets/val_references.jsonl')}
    out = ROOT / 'outputs/diagnostics/ocr_joining'
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in rows:
        sid = row['sequence_id']
        if sid not in candidates:
            continue
        for p, page in enumerate(candidates[sid]):
            groups = join_ocr_candidates(page)
            report = compare_groups_to_targets(groups, refs[sid][p])
            reports.append(dict(sequence_id=sid, page_index=p, raw_count=len(page), **report))
    save_jsonl(reports, out / 'allocations.jsonl')
    summary = dict(pages=len(reports), raw_fragments=sum(r['raw_count'] for r in reports),
                   proposed_groups=sum(len(r['groups']) for r in reports),
                   target_items=sum(len(refs[r['sequence_id']][r['page_index']]) for r in reports),
                   matched_targets=sum(g['target_index'] is not None for r in reports for g in r['groups']),
                   high_similarity_matches=sum((g['similarity'] or 0) >= .9 for r in reports for g in r['groups']),
                   note='Reference-assisted comparison only; grouping uses boxes alone. Group order is spatial, not panel-aware. Similarity matches do not establish correct bubble boundaries.')
    save_json(summary, out / 'summary.json')
    sample = rows[args.sample]
    sid, p = sample['sequence_id'], args.page
    report = next(r for r in reports if r['sequence_id'] == sid and r['page_index'] == p)
    prefix = f'sample_{args.sample}_page_{p}'
    with Image.open(image_paths(ROOT, sample)[p]) as image:
        raw, joined = draw_ocr_groups(image, candidates[sid][p], report['groups'])
    raw.save(out / f'{prefix}_raw.png')
    joined.save(out / f'{prefix}_groups.png')
    save_json(dict(**report, raw_ocr=candidates[sid][p]), out / f'{prefix}.json')
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
                    (g['id'], ', '.join(g['candidate_ids']), g['text'], g['target_index'], g['target_text'],
                     round(g['similarity'], 3) if g['similarity'] is not None else '')) + '</tr>' for g in report['groups'])
    document = f'''<!doctype html><meta charset="utf-8"><title>OCR joining preview</title>
<style>body{{font:15px system-ui;margin:24px}}.images{{display:flex;gap:20px}}figure{{margin:0;width:48%}}img{{width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}table{{border-collapse:collapse;width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(sid)} / page {p}</h1><p>Boxes-only grouping. Targets are used for comparison, not to produce text. Group IDs are spatial order; panel reading order is not inferred.</p>
<div class="images"><figure><figcaption>Raw OCR IDs</figcaption><img src="{prefix}_raw.png"></figure><figure><figcaption>Script allocation: one color per group</figcaption><img src="{prefix}_groups.png"></figure></div>
<h2>Script text allocation vs actual transcription</h2><table><tr><th>Group</th><th>OCR IDs</th><th>Script text</th><th>Target index</th><th>Actual transcription</th><th>Similarity</th></tr>{table}</table>
<h2>Unmatched targets</h2><pre>{html.escape(json.dumps(report['unmatched_targets'], indent=2, ensure_ascii=False))}</pre>
<h2>Raw OCR</h2><pre>{html.escape(json.dumps(candidates[sid][p], indent=2, ensure_ascii=False))}</pre>'''
    (out / f'{prefix}.html').write_text(document, encoding='utf-8')
    print(json.dumps(summary, indent=2))
    for g in report['groups']:
        print(g['id'], g['candidate_ids'], '=>', g['text'], '| target:', g['target_index'], '| similarity:', g['similarity'])
    print('Preview:', out / f'{prefix}.html')


if __name__ == '__main__':
    main()
