"""Export notebook predictions and run the unmodified TaskData/score.py scorer."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from TaskData import score


def score_saved_predictions(predictions_path, labels_path, sequence_ids, output_dir):
    """Score exactly the requested holdout; malformed outputs count as missing.

    No target repair or JSON-fence stripping is performed. Excluded predictions
    are recorded explicitly and penalized by the scorer's missing-row behavior.
    """
    ids = list(sequence_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Expected nonempty, unique evaluation sequence IDs')
    references = score.load_jsonl(labels_path, reference=True)
    unknown = set(ids) - references.keys()
    if unknown:
        raise ValueError(f'Unknown reference IDs: {sorted(unknown)}')
    references = {key: references[key] for key in ids}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    errors = []
    seen = set()
    candidate_path = output_dir / f'.candidate_{uuid.uuid4().hex}.jsonl'
    try:
        for line in Path(predictions_path).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record['sequence_id']
            if key not in references or key in seen:
                raise ValueError(f'Unexpected or duplicate prediction ID: {key}')
            seen.add(key)
            try:
                pages = json.loads(record['prediction'], object_pairs_hook=score.no_duplicate_keys)
                candidate_path.write_text(json.dumps({'sequence_id': key, 'pages': pages}, ensure_ascii=False) + '\n', encoding='utf-8')
                predictions.update(score.load_jsonl(candidate_path))
            except (ValueError, TypeError, KeyError) as error:
                errors.append({'sequence_id': key, 'error': str(error)})
    finally:
        candidate_path.unlink(missing_ok=True)
    submission = output_dir / 'submission.jsonl'
    submission.write_text(''.join(json.dumps({'sequence_id': key, 'pages': predictions[key]}, ensure_ascii=False) + '\n' for key in ids if key in predictions), encoding='utf-8')
    # Reload through the official loader, including UNKNOWN masking in references.
    gold = score.load_jsonl(labels_path)
    reference_path = output_dir / 'references.jsonl'
    reference_path.write_text(''.join(json.dumps({'sequence_id': key, 'pages': gold[key]}, ensure_ascii=False) + '\n' for key in ids), encoding='utf-8')
    report = score.evaluate(score.load_jsonl(reference_path, reference=True), score.load_jsonl(submission))
    (output_dir / 'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'invalid_predictions.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report
