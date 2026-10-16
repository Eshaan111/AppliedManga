"""Verify feature joins, split isolation, and crop preservation after preparation."""
from pathlib import Path
from collections import Counter
import json
import torch
from config import CONFIG, ROOT, BASELINE_ROOT
from bubble_features import sha256, save_json


def main():
    data = torch.load(CONFIG['cache_dir'] / 'samples.pt', map_location='cpu', weights_only=True)
    baseline = torch.load(BASELINE_ROOT / 'cache/samples.pt', map_location='cpu', weights_only=True)
    original = torch.load(BASELINE_ROOT / 'cache/magi.pt', map_location='cpu', weights_only=True)
    audit = json.loads((CONFIG['cache_dir'] / 'bubbles/development/text_feature_audit.json').read_text(encoding='utf-8'))
    audit = {(r['sequence_id'], r['box_id']): r for r in audit}
    parent_samples = {s['sequence_id']: s for s in baseline['samples']}
    coverage = Counter()
    assert len(data['samples']) == 80
    for sample in data['samples']:
        sid = sample['sequence_id']
        source = parent_samples[sid]
        assert len(sample['boxes']) == len(source['boxes'])
        for box, original_box in zip(sample['boxes'], source['boxes']):
            for key in ['box_id', 'bbox', 'page_index', 'text_index', 'label', 'gt_speaker']:
                assert box[key] == original_box[key], f'Original {key} changed'
            assert box['magi_ocr'] == original_box['ocr']
            assert box['ocr'] == box['ocr_selection']['selected_text']
        assert sample['text'].shape == (len(sample['boxes']), 768)
        assert torch.isfinite(sample['text']).all()
        assert torch.equal(sample['tail'], source['tail']), 'Full-page tail evidence changed'
        assert torch.equal(sample['geometry'], source['geometry'])
        n = len(sample['boxes'])
        assert sample['visual'].shape == (n, 256) and torch.isfinite(sample['visual']).all()
        for i, box in enumerate(sample['boxes']):
            row = audit[(sid, box['box_id'])]
            coverage[(int(box['label']), row['feature_source'])] += 1
            if row['bubble_id'] is None:
                assert torch.equal(sample['visual'][i], original['vectors'][sid][box['box_id']])
    split_report = {}
    hashes = []
    for name in ['train', 'val']:
        indices = data[name+'_indices']
        ids = {data['samples'][i]['sequence_id'] for i in indices}
        expected = {baseline['samples'][i]['sequence_id'] for i in baseline[name+'_indices']}
        assert ids == expected, 'Baseline split changed'
        split_report[name] = len(ids)
        hashes.append({p['sha256'] for i in indices for p in data['samples'][i]['pages']})
    assert not (hashes[0] & hashes[1]), 'Shared-page leakage'
    summaries = {}
    for split in ['development', 'test']:
        index = CONFIG['cache_dir'] / f'bubbles/{split}/index.json'
        if not index.is_file():
            continue
        rows = json.loads(index.read_text(encoding='utf-8'))
        assert len(rows) == (240 if split == 'development' else 3)
        counts = Counter()
        for row in rows:
            payload = torch.load(row['cache'], map_location='cpu', weights_only=True)
            assert payload['signature'] == row['signature']
            for bubble in payload['bubbles']:
                if bubble['kept']:
                    assert bubble['text_ids']
                    assert sha256(Path(row['cache']).parent / bubble['crop']) == bubble['crop_sha256']
                    vector = payload['embeddings'][bubble['bubble_id']]
                    assert vector.shape == (256,) and torch.isfinite(vector).all()
            counts.update({k:row[k] for k in ['candidates', 'kept', 'rejected', 'covered_texts', 'fallback_texts']})
        summaries[split] = dict(pages=len(rows), **counts)
    report = dict(status='passed', split=split_report, data=summaries,
        labeled_coverage=[dict(label=label, source=source, count=count)
                          for (label, source), count in sorted(coverage.items())])
    save_json(CONFIG['cache_dir'] / 'validation_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
