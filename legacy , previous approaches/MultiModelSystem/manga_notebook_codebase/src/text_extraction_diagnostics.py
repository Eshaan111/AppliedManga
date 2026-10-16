"""Evaluate saved TextExtractionPipeline results without altering extraction."""
import csv
import json
from pathlib import Path

from data_utils import load_jsonl, save_jsonl
from dataset_utils import save_json
from ocr_joining import compare_groups_to_targets
from ocr_reports import write_page_report, write_report_index
from ocr_utils import sha256
from text_extraction_pipeline import select_rows


def index_rows(records, label):
    result = {}
    for record in records:
        sid = record['sequence_id']
        if sid in result:
            raise ValueError(f'Duplicate sequence in {label}: {sid}')
        result[sid] = record
    return result


def load_inputs(root, candidates_path):
    root, path = Path(root).resolve(), Path(candidates_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Run TextExtractionPipeline.ipynb for val first: {path}')
    metadata_path, audit_path = path.with_suffix('.metadata.json'), path.parent / 'filter_audit.jsonl'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('split') != 'val':
        raise ValueError('Validation diagnostics require pipeline results from SPLIT="val"')
    if metadata.get('output_sha256') != sha256(path) or metadata.get('audit_sha256') != sha256(audit_path):
        raise ValueError('Pipeline candidates or filter audit changed; rerun extraction')
    if metadata.get('split_sha256') != sha256(root / 'outputs/split.json'):
        raise ValueError('Saved split changed since extraction')
    records = load_jsonl(path)
    candidates = index_rows(records, 'pipeline candidates')
    if not records or [r['sequence_id'] for r in records] != metadata['sequence_ids']:
        raise ValueError('Pipeline sequence IDs disagree with provenance')
    _, all_rows = select_rows(root, 'val')
    by_id = {r['sequence_id']: r for r in all_rows}
    if not set(candidates) <= set(by_id):
        raise ValueError('Pipeline results contain sequences outside validation')
    rows = [by_id[r['sequence_id']] for r in records]
    raw = {}
    for source in metadata['raw_sources']:
        raw_path = Path(source['path'])
        if sha256(raw_path) != source['sha256']:
            raise ValueError('Raw OCR source changed since extraction')
        raw.update(index_rows(load_jsonl(raw_path), 'raw OCR source'))
    audit = {}
    for page in load_jsonl(audit_path):
        key = (page['sequence_id'], page['page_index'])
        if key in audit:
            raise ValueError('Duplicate page in filter audit')
        audit[key] = page
    if set(audit) != {(sid, p) for sid in candidates for p in range(3)}:
        raise ValueError('Filter audit does not match pipeline pages')
    references_path = root / 'datasets/val_references.jsonl'
    references = index_rows(load_jsonl(references_path), 'validation references')
    for row in rows:
        sid = row['sequence_id']
        if [sha256(p) for p in row['images']] != metadata['images'].get(sid):
            raise ValueError(f'{sid}: images changed since extraction')
        for label, source in [('pipeline', candidates), ('raw OCR', raw), ('references', references)]:
            if sid not in source or not isinstance(source[sid].get('pages'), list) or len(source[sid]['pages']) != 3:
                raise ValueError(f'{sid}: missing or invalid pages in {label}')
        for p, groups in enumerate(candidates[sid]['pages']):
            raw_page = raw[sid]['pages'][p]
            raw_ids = [c['id'] for c in raw_page]
            if len(raw_ids) != len(set(raw_ids)):
                raise ValueError(f'{sid}: duplicate raw candidate ID')
            ids = [cid for group in groups for cid in group['candidate_ids']]
            excluded_ids = [e['candidate']['id'] for e in audit[sid, p]['excluded']]
            if (len(ids) != len(set(ids)) or set(ids) & set(excluded_ids)
                    or set(ids) | set(excluded_ids) != set(raw_ids)):
                raise ValueError(f'{sid}: groups and exclusions do not account for raw candidates')
    return rows, candidates, raw, audit, references, metadata, references_path, len(all_rows)


def run_diagnostics(root, candidates_path, output, *, render_previews=True):
    root, path, output = Path(root).resolve(), Path(candidates_path).resolve(), Path(output).resolve()
    if output == path.parent or output.is_relative_to(path.parent) or path.is_relative_to(output):
        raise ValueError('Diagnostics output must be separate from extraction results')
    rows, candidates, raw, audit, references, metadata, references_path, total = load_inputs(root, path)
    reports, table = [], []
    for row in rows:
        sid = row['sequence_id']
        for p, groups in enumerate(candidates[sid]['pages']):
            comparison = compare_groups_to_targets(groups, references[sid]['pages'][p])
            exclusions = audit[sid, p]['excluded']
            report = dict(sequence_id=sid, page_index=p, raw_count=len(raw[sid]['pages'][p]),
                          target_count=len(references[sid]['pages'][p]), filter_exclusions=exclusions,
                          excluded_sound_effects=[e['candidate'] for e in exclusions
                                                 if e['reason'] == 'sound_effect_word'],
                          processing_note='Saved TextExtractionPipeline groups, already filtered and joined. '
                                          'No OCR, filtering, or joining is run by diagnostics. '
                                          'References are comparison only; order is spatial.', **comparison)
            reports.append(report)
            for group in comparison['groups']:
                table.append(dict(sequence_id=sid, page_index=p,
                    status='matched' if group['target_index'] is not None else 'unmatched_group',
                    group_id=group['id'], candidate_ids=' '.join(group['candidate_ids']),
                    joined_text=group['text'], target_index=group['target_index'],
                    target_text=group['target_text'], similarity=group['similarity']))
            for target in comparison['unmatched_targets']:
                table.append(dict(sequence_id=sid, page_index=p, status='unmatched_target',
                                  group_id=None, candidate_ids=None, joined_text=None,
                                  target_index=target['target_index'], target_text=target['text'], similarity=None))
    group_count, target_count = sum(len(r['groups']) for r in reports), sum(r['target_count'] for r in reports)
    matches = [g for r in reports for g in r['groups'] if g['target_index'] is not None]
    summary = dict(sequences=len(rows), validation_sequences_total=total, pages=len(reports),
        raw_fragments=sum(r['raw_count'] for r in reports), proposed_groups=group_count,
        excluded_sound_effect_fragments=sum(len(r['excluded_sound_effects']) for r in reports),
        excluded_fragments=sum(len(r['filter_exclusions']) for r in reports),
        target_items=target_count, matched_targets=len(matches),
        unmatched_groups=group_count-len(matches), unmatched_targets=target_count-len(matches),
        high_similarity_matches=sum(g['similarity'] >= .9 for g in matches),
        matched_target_fraction=len(matches)/target_count if target_count else None,
        mean_matched_similarity=sum(g['similarity'] for g in matches)/len(matches) if matches else None,
        matching_threshold=.5, high_similarity_threshold=.9,
        input_candidates=str(path), input_candidates_sha256=sha256(path),
        references_sha256=sha256(references_path),
        pipeline_metadata_sha256=sha256(path.with_suffix('.metadata.json')),
        diagnostics_code_sha256=sha256(Path(__file__)),
        note='Evaluates saved pipeline groups unchanged. Fuzzy matching ignores punctuation and spacing. '
             'These diagnostics do not verify bubble boundaries, speakers, or manga reading order.')
    output.mkdir(parents=True, exist_ok=True)
    # Remove the obsolete output from the previous re-joining workflow.
    (output / 'joined.jsonl').unlink(missing_ok=True)
    save_jsonl(reports, output / 'allocations.jsonl')
    save_json(summary, output / 'summary.json')
    fields = ['sequence_id', 'page_index', 'status', 'group_id', 'candidate_ids',
              'joined_text', 'target_index', 'target_text', 'similarity']
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    entries = []
    if render_previews:
        for sample, row in enumerate(rows):
            for p in range(3):
                entries.append(write_page_report(root, row, sample, reports[sample*3+p],
                                                  raw[row['sequence_id']]['pages'][p], output))
        write_report_index(entries, output)
    (output / 'transcripts.txt').write_text('\n\n'.join(
        f"{r['sequence_id']} / page {r['page_index']}\n" + '\n'.join(g['text'] for g in r['groups'])
        for r in reports) + '\n', encoding='utf-8')
    return summary
