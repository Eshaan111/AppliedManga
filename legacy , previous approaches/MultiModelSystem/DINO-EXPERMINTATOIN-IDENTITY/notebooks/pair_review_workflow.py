"""Reusable code for notebook 04: blind human review, sampling, and score analysis.

No model runs at import time. Crop identities are scoped by their embedding run.
"""
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

TAGS = ('pose_expression', 'lookalike', 'small', 'profile', 'occluded')
THRESHOLD = 0.8


def pair_key(row):
    return (row['run_id'], row['sequence_id'], row.get('strategy', 'face'),
            *sorted((row['crop1'], row['crop2'])))


def load_run(run):
    """Join saved scores to detection page context, without altering source files."""
    run = Path(run).resolve()
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    manifest_path = Path(config['detection_manifest'])
    if not manifest_path.is_absolute():
        manifest_path = run / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pages = {(p['sequence_id'], str(p['page_index'])): p for p in manifest['pages']}
    with (run / 'pair_review.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        row = dict(row, run_id=str(run))
        score = float(row['similarity'])
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('Invalid cosine similarity')
        for side in (1, 2):
            page = pages[row['sequence_id'], str(row[f'page{side}'])]
            row[f'page_path{side}'] = page['image_path']
            row[f'overlay_path{side}'] = page['overlay_path']
        row['split'] = pages[row['sequence_id'], str(row['page1'])].get('split', 'unknown')
        for tag in TAGS:
            row.setdefault(tag, '')
        result.append(row)
    validate_rows(result)
    return result


def validate_rows(rows):
    seen = set()
    for row in rows:
        key = pair_key(row)
        if key in seen:
            raise ValueError(f'Duplicate pair: {key}')
        seen.add(key)
        if row.get('same_character', '') not in ('', 'same', 'different'):
            raise ValueError('same_character must be same, different, or blank')
        for tag in TAGS:
            if row.get(tag, '') not in ('', 'yes'):
                raise ValueError(f'{tag} must be yes or blank')
        score = float(row['similarity'])
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('Invalid cosine similarity')


def sample_pairs(rows, target=75, seed=42, sequence_ids=None):
    """Score-independent shuffled round-robin sampling across sequences.

    This provides candidates, not guaranteed identity classes or difficult poses.
    Review tags and add targeted examples after inspecting the manga pages.
    """
    if target < 1:
        raise ValueError('target must be positive')
    validate_rows(rows)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=pair_key):
        if sequence_ids is None or row['sequence_id'] in sequence_ids:
            groups[row['sequence_id']].append(dict(row))
    ids = sorted(groups)
    rng.shuffle(ids)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < target and any(groups.values()):
        for sid in ids:
            if groups[sid] and len(selected) < target:
                selected.append(groups[sid].pop())
    return selected


def write_csv(rows, path):
    """Write a new review file; never silently replace human labels."""
    rows = list(rows)
    validate_rows(rows)
    if not rows:
        raise ValueError('No candidate pairs; inspect detections or select more pages')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows)
    return rows


def import_labels(candidate_csv, downloaded_csv, output_csv):
    """Import only human columns, matching exact run-scoped pairs.

    Saved similarities and crop/page provenance come from candidate_csv, not
    editable downloads. The labeled result goes to a new file.
    """
    originals = read_csv(candidate_csv)
    edited = read_csv(downloaded_csv)
    by_key = {pair_key(row): row for row in edited}
    if set(by_key) != {pair_key(row) for row in originals}:
        raise ValueError('Downloaded pairs do not match this review batch')
    for row in originals:
        for field in ('same_character', 'notes', *TAGS):
            row[field] = by_key[pair_key(row)].get(field, '')
    return write_csv(originals, output_csv)


def write_blind_html(rows, path):
    """Local HTML review: label first, then reveal score; download edited CSV.

    Browser cannot write the original file. Download and explicitly import labels.
    Open this HTML directly from disk so file:// image paths can be displayed.
    """
    rows = list(rows)
    validate_rows(rows)
    if not rows:
        raise ValueError('No pairs to review')
    cards = []
    for index, row in enumerate(rows):
        figures = []
        for side in (1, 2):
            crop = html.escape(Path(row[f'crop_path{side}']).resolve().as_uri(), quote=True)
            page = html.escape(Path(row[f'page_path{side}']).resolve().as_uri(), quote=True)
            overlay = html.escape(Path(row[f'overlay_path{side}']).resolve().as_uri(), quote=True)
            caption = html.escape(f"page {row[f'page{side}']} / {row[f'crop{side}']}")
            figures.append(f'<figure><img class="crop" src="{crop}"><figcaption>{caption}</figcaption>'
                           f'<details><summary>Full manga page and crop locations</summary>'
                           f'<a href="{page}" target="_blank">Open original page</a>'
                           f'<a href="{overlay}" target="_blank"><img class="page" src="{overlay}"></a>'
                           '</details></figure>')
        checks = ''.join(f'<label><input type="checkbox" data-tag="{tag}" '
                         f'{"checked" if row.get(tag) == "yes" else ""}>{tag}</label> ' for tag in TAGS)
        label = row.get('same_character', '')
        options = ''.join(f'<option value="{value}" {"selected" if label == value else ""}>{text}</option>'
                          for value, text in (('', 'Uncertain / unreviewed'), ('same', 'Same'), ('different', 'Different')))
        cards.append(f'<article data-index="{index}"><h2>Pair {index + 1}: '
                     f'{html.escape(row["sequence_id"])}</h2><div class="faces">{"".join(figures)}</div>'
                     f'<p>Identity from manga pages: <select>{options}</select></p><p>{checks}</p>'
                     f'<textarea placeholder="Identity evidence; explain uncertainty or detector errors">'
                     f'{html.escape(row.get("notes", ""))}</textarea>'
                     '<p><button class="reveal">Compare with score after labeling</button> '
                     '<span class="score"></span></p></article>')
    payload = json.dumps(rows, ensure_ascii=True).replace('<', '\\u003c')
    document = '''<!doctype html><meta charset="utf-8"><title>Blind manga identity review</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}article{background:white;padding:18px;margin:20px 0}
.faces{display:flex;gap:24px}figure{margin:0;flex:1;min-width:0}.crop{width:220px;height:220px;object-fit:contain}
.page{width:100%}textarea{width:95%;height:65px}header{position:sticky;top:0;background:#fff;padding:12px;z-index:1}
button,select{padding:8px}label{display:inline-block;margin:5px}a{display:block}</style>
<header><b>Judge identity from the manga page before revealing cosine similarity.</b>
<p>Uncertain pairs stay blank; explain in notes. Tags describe visual difficulty, not model predictions.</p>
<button id="download">Download labeled CSV</button> <span id="progress"></span>
<p>Edits stay on this page until you download. Download before closing or reloading. Import the CSV in notebook 04.</p></header>
''' + ''.join(cards) + '<script>const rows=' + payload + ''';
function progress(){document.querySelector('#progress').textContent=rows.filter(r=>r.same_character).length+' / '+rows.length+' labeled';}
document.querySelectorAll('article').forEach(card=>{
 const row=rows[Number(card.dataset.index)];
 card.querySelector('select').addEventListener('change',e=>{row.same_character=e.target.value;card.querySelector('.score').textContent='';progress();});
 card.querySelector('textarea').addEventListener('input',e=>row.notes=e.target.value);
 card.querySelectorAll('[data-tag]').forEach(box=>box.addEventListener('change',()=>row[box.dataset.tag]=box.checked?'yes':''));
 card.querySelector('.reveal').addEventListener('click',()=>{
  card.querySelector('.score').textContent=row.same_character?
   'Cosine '+Number(row.similarity).toFixed(4)+' | at 0.8: '+(Number(row.similarity)>=0.8?'probably same':'probably different'):
   'Leave uncertain cases blank. Record a confident identity judgment before revealing.';
 });
});
document.querySelector('#download').addEventListener('click',()=>{
 const fields=Object.keys(rows[0]);
 const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 const csv=[fields.map(quote).join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\\r\\n');
 const url=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='pair_review_labeled.csv';a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});progress();</script>'''
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(document)
    return path


def coverage(rows):
    """Show confirmed labels and difficulty coverage; blanks are not negatives."""
    labeled = [r for r in rows if r.get('same_character') in ('same', 'different')]
    return dict(candidates=len(rows), labeled=len(labeled), uncertain_or_unreviewed=len(rows)-len(labeled),
                labels=dict(Counter(r['same_character'] for r in labeled)),
                sequences=dict(Counter(r['sequence_id'] for r in labeled)),
                splits=dict(Counter(r.get('split', 'unknown') for r in labeled)),
                difficulty={tag: dict(Counter(r['same_character'] for r in labeled if r.get(tag) == 'yes'))
                            for tag in TAGS})


def analyze_scores(rows, output):
    """Descriptive distributions and fixed-0.8 confusion counts; no tuning."""
    import numpy as np
    import matplotlib.pyplot as plt

    validate_rows(rows)
    labeled = [r for r in rows if r.get('same_character') in ('same', 'different')]
    scores = {label: np.array([float(r['similarity']) for r in labeled if r['same_character'] == label])
              for label in ('same', 'different')}
    if any(len(v) == 0 for v in scores.values()):
        raise ValueError('Label at least one same and one different pair before analysis')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new analysis directory: {output}')
    output.mkdir(parents=True)
    stats = {label: dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                        minimum=float(v.min()), maximum=float(v.max()),
                        q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)))
             for label, v in scores.items()}
    confusion = dict(tp=sum(r['same_character'] == 'same' and float(r['similarity']) >= THRESHOLD for r in labeled),
                     fn=sum(r['same_character'] == 'same' and float(r['similarity']) < THRESHOLD for r in labeled),
                     fp=sum(r['same_character'] == 'different' and float(r['similarity']) >= THRESHOLD for r in labeled),
                     tn=sum(r['same_character'] == 'different' and float(r['similarity']) < THRESHOLD for r in labeled))
    # Empirical P(score_same > score_different), counting ties as half.
    comparisons = scores['same'][:, None] - scores['different'][None, :]
    auc = float((np.sum(comparisons > 0) + .5 * np.sum(comparisons == 0)) / comparisons.size)
    bins = np.linspace(-1, 1, 21)
    hist = {label: np.histogram(v, bins=bins)[0] / len(v) for label, v in scores.items()}
    overlap = float(np.minimum(hist['same'], hist['different']).sum())
    report = dict(coverage=coverage(rows), descriptive_statistics=stats, fixed_threshold=THRESHOLD,
                  confusion=confusion, empirical_auc=auc, histogram_overlap=overlap,
                  overlap_bins=bins.tolist(),
                  limitations='Exploratory selected pairs, correlated within pages/sequences; no threshold tuning or population claim.')
    (output / 'score_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, values in scores.items():
        axes[0].hist(values, bins=bins, weights=np.ones(len(values))/len(values), alpha=.55, label=f'{label} (n={len(values)})')
        ordered = np.sort(values)
        axes[1].step(ordered, np.arange(1, len(values)+1)/len(values), where='post', label=label)
    for ax in axes:
        ax.axvline(THRESHOLD, color='black', linestyle='--', label='fixed 0.8')
        ax.set(xlim=(-1, 1), xlabel='DINO cosine similarity')
        ax.legend()
    axes[0].set(ylabel='Fraction of each class', title='Same vs different')
    axes[1].set(ylabel='Empirical cumulative fraction', title='Score distributions')
    fig.tight_layout()
    fig.savefig(output / 'same_vs_different.png', dpi=160)
    plt.close(fig)
    disagreements = [r for r in labeled if (float(r['similarity']) >= THRESHOLD) != (r['same_character'] == 'same')]
    if disagreements:
        write_csv(disagreements, output / 'threshold_disagreements.csv')
    return report
