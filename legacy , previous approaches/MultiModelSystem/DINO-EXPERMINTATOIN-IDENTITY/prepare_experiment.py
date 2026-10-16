"""Prepare five training-only sequences, a portable annotator, and the notebook."""
import base64
import json
from pathlib import Path

from identity import ROOT, save_json


def prepare_annotations():
    (ROOT / 'character_crops').mkdir(exist_ok=True)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    task = ROOT.parent / 'manga_notebook_codebase/TaskData'
    split = json.loads((task.parent / 'outputs/split.json').read_text())
    manifest = {s['sequence_id']: s for s in json.loads((task / 'sequences.json').read_text())}
    sequences = []
    for n, sid in enumerate(split['train'][:5]):
        seq = manifest[sid]
        sequences.append(dict(sequence_id=sid, split='calibration' if n < 3 else 'heldout',
                              pages=[dict(page_index=i, image=image, characters=[]) for i, image in enumerate(seq['images'])]))
    path = ROOT / 'annotations/characters.json'
    if not path.exists():
        save_json(path, dict(schema_version=1, task_data_root='../manga_notebook_codebase/TaskData', sequences=sequences))
    data = json.loads(path.read_text())
    assets = {}
    task_root = (ROOT / data['task_data_root']).resolve()
    for seq in data['sequences']:
        for page in seq['pages']:
            path = task_root / page['image']
            assets[page['image']] = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    template = (ROOT / 'annotator_template.html').read_text(encoding='utf-8')
    template = template.replace('__DATA__', json.dumps(data).replace('<', '\\u003c'))
    template = template.replace('__ASSETS__', json.dumps(assets))
    (ROOT / 'annotations/annotate.html').write_text(template, encoding='utf-8')
    labels = {s['sequence_id']: {c['crop_id']: c.get('identity_label') for p in s['pages'] for c in p['characters']} for s in data['sequences']}
    save_json(ROOT / 'annotations/character_identity_labels.json', labels)
    print('Prepared annotations/annotate.html; no boxes or identity labels are invented.')


def prepare_notebook():
    import nbformat as nb
    notebook = nb.v4.new_notebook()
    cells = []
    def md(text):
        cells.append(nb.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nb.v4.new_code_cell(text))
    md('# DINOv2-small: manual character identity experiment\n\n'
       'Keep this separate from OCR. Use original page pixel coordinates and visually reviewed anonymous identities. '
       'The starter set has five sequences from the existing training split: three calibration, two heldout. '
       'The project validation/test sets are untouched. These are candidate sequences; replace them if necessary to cover pose and appearance variation. '
       'Aim for 50–100 appearances across pages. Draw face, head + hair, and upper-body boxes for the same appearances. '
       'Every reported pair stays within one sequence and one crop strategy and crosses pages. '
       'Heldout results are independent only if neither their labels nor failures guide threshold or crop-strategy selection.')
    md('## 1. Load DINO\nRun `download_model.py` first if the local model is missing. Select the existing **AppliedTask (CUDA GPU)** kernel.')
    code("from pathlib import Path\nimport sys\nimport json\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, FileLink\nROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\nif ROOT.name != 'DINO-EXPERMINTATOIN-IDENTITY':\n    ROOT = ROOT / 'DINO-EXPERMINTATOIN-IDENTITY'\nassert (ROOT / 'identity.py').exists(), ROOT\nsys.path.insert(0, str(ROOT))\nimport identity as dino\nprocessor, model, device = dino.load_model()\nprint(device, model.config.hidden_size)")
    md('## 2. Load manually cropped characters\nOpen `annotations/annotate.html` in your browser. '
       'Drag a character box, enter an anonymous identity such as A/B/C, and save each of the three strategies under the same appearance ID. '
       'Labels restart within every sequence. Export and replace `annotations/characters.json` with the downloaded file, then run this cell. '
       'Missing strategies are allowed, but fair strategy comparison requires matching appearances. Leave identity blank for ambiguous appearances; those are excluded from labelled pair evaluation.')
    code("annotation_path = ROOT / 'annotations/characters.json'\ndata, rows = dino.load_annotations(annotation_path)\nassert rows, 'Open annotations/annotate.html and annotate characters first.'\nrows = dino.export_crops(rows)\nprint(len(rows), 'crop variants from', len({(r['sequence_id'], r['crop_id']) for r in rows}), 'appearances')\ndisplay(FileLink(str(ROOT / 'annotations/annotate.html')))")
    md('## 3. Generate normalized embeddings')
    code("vectors = dino.embed(rows, processor, model, device)\nprint(vectors.shape)\nprint('L2 norms:', np.linalg.norm(vectors, axis=1).min(), np.linalg.norm(vectors, axis=1).max())")
    md('## 4. Calculate similarity matrices')
    code("group_indices = dino.groups(rows)\nsimilarity_matrices = {key: np.clip(vectors[idx] @ vectors[idx].T, -1, 1) for key, idx in group_indices.items()}\nlist(similarity_matrices)")
    md('## 5. Display a similarity matrix')
    code("first_key = next(iter(group_indices))\ndino.plot_matrix(rows, vectors, group_indices[first_key])\nplt.show()")
    md('## 6. Build SAME / DIFFERENT cross-page pairs')
    code("pairs = dino.make_pairs(rows, vectors)\nprint('Cross-page labelled pairs:', len(pairs))\nfor strategy in dino.STRATEGIES:\n    subset = [p for p in pairs if p['strategy'] == strategy]\n    print(strategy, len(subset), 'pairs;', sum(p['same_character'] for p in subset), 'same')")
    md('## 7. Plot calibration distributions')
    code("calibrations = {}\nfor strategy in dino.STRATEGIES:\n    subset = [p for p in pairs if p['strategy'] == strategy and p['split'] == 'calibration']\n    if {p['same_character'] for p in subset} != {True, False}:\n        print(strategy, 'needs labelled same and different cross-page calibration pairs')\n        continue\n    calibrations[strategy] = dino.calibrate(subset)\n    dino.plot_distributions(subset, calibrations[strategy]['threshold'])\n    plt.title(strategy + ' — calibration only')\n    plt.show()")
    md('## 8. Calibrate thresholds and evaluate heldout pairs\nThresholds maximize balanced accuracy on calibration sequences only. '
       'Calibration accuracy is a fit score, not a final benchmark. Do not choose a strategy or threshold by heldout performance. '
       'Cosine scores span −1 to 1. Different pairs can outnumber same pairs, so report balanced accuracy alongside ordinary accuracy.')
    code("for strategy, report in calibrations.items():\n    heldout = [p for p in pairs if p['strategy'] == strategy and p['split'] == 'heldout']\n    print(strategy, 'threshold', report['threshold'])\n    print('CALIBRATION', report['calibration'])\n    print('HELDOUT', dino.metrics(heldout, report['threshold']))")
    md('## 9. Cluster within each sequence\nGreedy clustering assigns each crop to a normalized mean centroid or creates a new identity. '
       'Identities restart per sequence and strategy; assignments are order-sensitive. Pairwise threshold accuracy does not prove clustering accuracy.')
    code("assignments = {}\nfor strategy, report in calibrations.items():\n    assignments.update({i: label for i, label in dino.cluster(rows, vectors, report['threshold']).items() if rows[i]['strategy'] == strategy})\n[(rows[i]['sequence_id'], rows[i]['strategy'], rows[i]['crop_id'], label) for i, label in assignments.items()][:20]")
    md('## 10. Save results and inspect failures\nThis reruns extraction to produce a self-contained run directory: embeddings, matrices, '
       'pair scores, metrics, failure image grids, annotation snapshot, and separate `character_results.jsonl`. '
       'Results contain no OCR text or evaluation identity labels. Use calibration failures to improve the experiment; freeze decisions before inspecting heldout failures.')
    code("from datetime import datetime\noutput = ROOT / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S')\nreports = dino.run_experiment(annotation_path, output)\nprint(json.dumps(reports, indent=2))\nprint('Saved to', output)\nfrom IPython.display import Image as DisplayImage\nfor path in sorted(output.glob('*_calibration_failures.png')):\n    display(DisplayImage(filename=str(path)))")
    notebook.cells = cells
    notebook.metadata = dict(kernelspec=dict(display_name='AppliedTask (CUDA GPU)', language='python', name='venvappliedtask'),
                             language_info=dict(name='python'))
    path = ROOT / 'notebooks/02_character_identity_dino.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        nb.write(notebook, path)
    print(path)


if __name__ == '__main__':
    prepare_annotations()
    prepare_notebook()
