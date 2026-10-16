"""Regenerate notebook 04 without executing models or labeling any pairs."""
import json
from pathlib import Path


def build():
    cells = []

    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(keepends=True),
                          execution_count=None, outputs=[]))

    md('''# 3. Review DINO face pairs and compare human identity labels

This notebook contains the code that produces the review and analysis outputs. It does **not** supply identity judgments or claim that DINO works. Start with the seven saved test pairs, then collect roughly **50–100 confident labels across several sequences**.

Our workflow is: manga pages → hysts YOLOv3 face detections → exact face crops → DINOv2-small normalized embeddings → cross-page cosine scores → human identity review → same/different score distributions. Notebook `02_character_detection.ipynb` handles detection; `03_dino_identity.ipynb` handles embeddings. This notebook reuses their saved artifacts and can run a larger batch through the same code.

Judge identity from the manga page and visual character features first. Scores are hidden in the new review view until you enter `same` or `different`. Leave uncertain cases blank and explain them in `notes`. Cosine similarity is not a probability. Keep **0.8 unchanged**; this is a diagnostic comparison, not threshold calibration.

The reusable implementation lives beside this notebook in `pair_review_workflow.py`. Models only load when you explicitly run the expansion cell. Manual box annotations can continue independently: we read only page selection from `characters.json`, never its boxes or character labels. Automatic crop IDs are scoped to their detection/embedding run.''')
    code('''from pathlib import Path
from datetime import datetime
import sys, json
from IPython.display import display, Image, FileLink

# Works from the workspace root, experiment root, or notebooks directory.
HERE = Path.cwd().resolve()
ROOT = next((p if (p / 'automatic_identity.py').is_file()
             else p / 'DINO-EXPERMINTATOIN-IDENTITY'
             for p in (HERE, *HERE.parents)
             if (p / 'automatic_identity.py').is_file()
             or (p / 'DINO-EXPERMINTATOIN-IDENTITY' / 'automatic_identity.py').is_file()), None)
assert ROOT is not None, 'Open this notebook inside the MultiModelSystem workspace'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks'))
import pair_review_workflow as review

THRESHOLD = 0.8  # Do not tune it in this workflow.
assert review.THRESHOLD == THRESHOLD
SESSION = ROOT / 'outputs' / f'manual_pair_review_{datetime.now():%Y%m%d_%H%M%S_%f}'
SESSION.mkdir(parents=True, exist_ok=False)
print('New review session:', SESSION)
''')
    md('''## Start with the saved seven-pair smoke test

The existing `outputs/two_model_pairs/pair_review.html` and `pair_review.csv` are the starting artifacts. The old HTML already exposes scores and sorts by them; use the new **blind review HTML** generated below to label without score hints. We preserve the original CSV and create a separate review copy, including full-page overlays, source-page links, notes, and difficulty tags. The notebook checks that the expected seven-pair run is present.

Open the generated HTML **directly from disk in your browser** (paste its printed absolute path). A notebook-server preview may block local image paths. Expand “Full manga page and crop locations” to inspect the page; click the overlay to enlarge it. An overlay shows all detected crop IDs, so match the ID printed below the face to its location. Use context and persistent character features; pose, expression, or a detector error alone is not identity evidence.

Choose `same` or `different`, write evidence in notes, and mark applicable difficulty tags. Uncertain pairs stay blank, with notes explaining why. Only then click the score-comparison button. Do not revise a label just because it disagrees with the score. If page inspection reveals a real mistake, document the correction.

Click **Download labeled CSV before closing or reloading**. Browser edits are not written directly to the filesystem. Import that download in the later cell; the original run is never overwritten.''')
    code('''SMOKE_RUN = ROOT / 'outputs' / 'two_model_pairs'
smoke_pairs = review.load_run(SMOKE_RUN)
assert len(smoke_pairs) == 7, f'Expected seven test pairs, found {len(smoke_pairs)}'
# Shuffle without consulting similarity; load_run itself does not display scores.
smoke_pairs = review.sample_pairs(smoke_pairs, target=7, seed=42)
SMOKE_CSV = review.write_csv(smoke_pairs, SESSION / 'smoke_candidates.csv')
SMOKE_HTML = review.write_blind_html(smoke_pairs, SESSION / 'smoke_blind_review.html')
print('Original artifacts:', SMOKE_RUN / 'pair_review.html', SMOKE_RUN / 'pair_review.csv', sep='\\n')
print('Open this local file:', SMOKE_HTML)
display(FileLink(str(SMOKE_HTML)))
''')
    md('''## Expand across sequences using the same two models

The seven pairs only check that the workflow runs. They come from one sequence marked heldout, so keep their diagnostic labels separate from the new calibration batch; do not use them to choose a threshold.

The expansion code selects up to five **calibration** sequences and up to five pages per sequence. Unlike a global page cap, this makes room for several sequences. Increase these limits if fewer than 100 candidates are detected. Only existing selected pages are used; if more are needed, extend the page selection separately. Missing detections cannot be assessed as embedding failures: first inspect overlays for missed faces and false positives.

Set `RUN_EXPANSION = True` to run local YOLOv3 and DINO inference with the same settings as the smoke test. Use the AppliedTask CUDA kernel and previously downloaded weights. No installation or model download is performed here. Alternatively set `EXPANDED_RUN` to an existing embedding run that contains several sequences, leaving the flag false. Fresh output directories protect prior artifacts.''')
    code('''RUN_EXPANSION = False  # Set True when ready to perform local model inference.
EXPANDED_RUN = None     # Or Path(r'C:/.../outputs/pairs_multi_sequence') to reuse a run.
N_SEQUENCES = 5
PAGES_PER_SEQUENCE = 5

if RUN_EXPANSION:
    import detection
    import automatic_identity as auto
    from collections import defaultdict
    # select_pages reads page paths only; no manual boxes or identity labels.
    all_pages = detection.select_pages(limit=1_000_000)
    grouped = defaultdict(list)
    for page in all_pages:
        if page.get('split') == 'calibration':
            grouped[page['sequence_id']].append(page)
    chosen_ids = sorted(grouped)[:N_SEQUENCES]
    assert len(chosen_ids) >= 3, 'Select at least three calibration sequences first'
    selected_pages = [p for sid in chosen_ids for p in grouped[sid][:PAGES_PER_SEQUENCE]]
    smoke_config = json.loads((SMOKE_RUN / 'run_config.json').read_text(encoding='utf-8'))
    confidence = smoke_config['detector_config']['confidence']
    detection_output = SESSION / 'expanded_faces'
    manifest = detection.run_detection(selected_pages, detection_output, confidence=confidence)
    for page in manifest['pages']:
        print(page['sequence_id'], 'page', page['page_index'])
        display(Image(filename=page['overlay_path'], width=800))
    EXPANDED_RUN = SESSION / 'expanded_pairs'
    auto.run_identity(detection_output / 'detections.json', EXPANDED_RUN,
                      model_path=Path(smoke_config['model_path']), threshold=THRESHOLD,
                      max_pages=PAGES_PER_SEQUENCE)
else:
    print('Expansion inference is off. Set RUN_EXPANSION=True or provide EXPANDED_RUN.')
''')
    md('''## Prepare a score-independent review batch

We select **100 candidate pairs** with seeded, shuffled round-robin sampling across sequences, aiming for about **75 confident labels** after uncertain cases are excluded. Sampling never uses cosine similarity or model predictions. It cannot guarantee class balance or difficult poses because those require human review.

Include same characters with changed pose/expression, different characters with similar appearance, and small/profile/partially hidden faces. Record those with `pose_expression`, `lookalike`, `small`, `profile`, and `occluded` tags. A tag means at least one face has that property. Use `lookalike` for visually similar **different** characters; explain the distinguishing evidence. A detector false positive should remain blank with a note, rather than becoming a different-character label.

If confident labels or difficult examples are scarce, expand pages/sequences, or use the supplemental cell below to inspect remaining pairs. Count labels after review, not generated candidates. Do not select extra examples by score. The target is a useful diagnostic sample, not a claim of population accuracy.''')
    code('''BATCH_CSV = BATCH_HTML = None
all_candidates = batch = []
if EXPANDED_RUN is not None:
    all_candidates = review.load_run(Path(EXPANDED_RUN))
    # Preserve heldout sequences for a separate future evaluation.
    all_candidates = [r for r in all_candidates if r['split'] == 'calibration']
    batch = review.sample_pairs(all_candidates, target=100, seed=42)
    assert batch, 'No calibration cross-page pairs; inspect detections and selected pages'
    assert len({r['sequence_id'] for r in batch}) >= 3, 'Need pairs from at least three sequences'
    BATCH_CSV = review.write_csv(batch, SESSION / 'expanded_candidates.csv')
    BATCH_HTML = review.write_blind_html(batch, SESSION / 'expanded_blind_review.html')
    print('Candidate coverage:', review.coverage(batch))  # No similarity scores.
    if len(batch) < 100:
        print('Candidate shortage: select more pages/sequences; do not duplicate pairs.')
    print('Open this local file:', BATCH_HTML)
    display(FileLink(str(BATCH_HTML)))
else:
    print('Run or reuse an expanded embedding batch first.')
''')
    md('''## Import the human review CSV safely

Set the download paths below after each browser review. The import joins exact `(run, sequence, strategy, crop pair)` keys and copies only `same_character`, `notes`, and difficulty tags. Original scores and provenance are retained. Invalid labels, duplicate pairs, or a download from another batch fail explicitly. Human identity labels never feed back into embeddings.

You can also edit the candidate CSV directly in a spreadsheet or text editor: preserve all IDs and scores; use only `same`, `different`, or blank in `same_character`; use `yes` or blank in tag columns. Point the download variable at that edited file. Notes may contain commas, quotes, and newlines; standard CSV quoting is supported. Each import writes a new labeled file so rerunning cannot silently overwrite previous review.''')
    code('''SMOKE_DOWNLOAD = None  # Path(r'C:/Users/ESHAAN/Downloads/pair_review_labeled.csv')
BATCH_DOWNLOAD = None  # Path to the separately downloaded expanded review.
SMOKE_LABELED = BATCH_LABELED = None
if SMOKE_DOWNLOAD is not None:
    SMOKE_LABELED = SESSION / f'smoke_labeled_{datetime.now():%H%M%S_%f}.csv'
    review.import_labels(SMOKE_CSV, SMOKE_DOWNLOAD, SMOKE_LABELED)
    print('Smoke review coverage:', review.coverage(review.read_csv(SMOKE_LABELED)))
if BATCH_DOWNLOAD is not None:
    assert BATCH_CSV is not None, 'Prepare an expanded batch before importing its labels'
    BATCH_LABELED = SESSION / f'expanded_labeled_{datetime.now():%H%M%S_%f}.csv'
    review.import_labels(BATCH_CSV, BATCH_DOWNLOAD, BATCH_LABELED)
    labeled_rows = review.read_csv(BATCH_LABELED)
    status = review.coverage(labeled_rows)
    print(json.dumps(status, indent=2))
    if not 50 <= status['labeled'] <= 100 or len(status['sequences']) < 3:
        print('Continue review toward 50–100 confident labels across at least three sequences.')
    print('Check difficulty counts: changed poses for same, lookalikes for different, and small/profile/occluded faces.')
''')
    md('''## Optional supplemental candidates

Enable this if the first batch leaves too many uncertain pairs or lacks the required visual cases. The next batch contains unused pairs from the **same expanded run**, still selected without scores. Inspect their pages and label the needed cases with notes; leave uncertain cases blank. If the pool is exhausted, run a larger expansion and review that as a separate batch. Crop IDs alone must never be merged across runs.

Download and import this review using its own candidate CSV. Merge the labeled rows with the initial labeled rows only after validation; the code below does this before the analysis. Difficult examples selected by a reviewer make this a diagnostic sample, and can shift the apparent score distributions.''')
    code('''MAKE_SUPPLEMENT = False
SUPPLEMENT_DOWNLOAD = None
SUPPLEMENT_LABELED = None
if MAKE_SUPPLEMENT:
    used = {review.pair_key(r) for r in batch}
    remaining = [r for r in all_candidates if review.pair_key(r) not in used]
    supplement = review.sample_pairs(remaining, target=40, seed=43)
    if supplement:
        SUPPLEMENT_CSV = review.write_csv(supplement, SESSION / 'supplement_candidates.csv')
        path = review.write_blind_html(supplement, SESSION / 'supplement_blind_review.html')
        print('Open this local file:', path)
    else:
        print('No unused candidates: expand pages/sequences in a new session.')
if SUPPLEMENT_DOWNLOAD is not None:
    assert 'SUPPLEMENT_CSV' in globals(), 'Generate a supplemental review first'
    SUPPLEMENT_LABELED = SESSION / f'supplement_labeled_{datetime.now():%H%M%S_%f}.csv'
    review.import_labels(SUPPLEMENT_CSV, SUPPLEMENT_DOWNLOAD, SUPPLEMENT_LABELED)
''')
    md('''## Compare scores only after human labels are saved

The analysis ignores blanks. It requires both identity classes. It saves a JSON summary, a normalized histogram and empirical cumulative distribution plot, and a CSV of disagreements with the unchanged **0.8** suggestion (if any). Histograms use identical fixed bins. Histogram overlap ranges from 0 (no overlapping bin mass) to 1 (identical bin mass); it depends on bin width. Empirical AUC is the fraction of same/different score comparisons ordered correctly, with half credit for ties. Neither statistic tunes a threshold.

Read the plot together with sample counts, per-sequence coverage, and difficulty tags. Promising separation means same pairs generally score higher than different pairs, including the difficult cases. Substantial overlap, high-scoring lookalikes, or low-scoring same characters suggests reconsidering the embedding model or crop strategy. There is no invented pass/fail cutoff. Inspect disagreement pages to distinguish mistaken labels, detection/crop problems, and embedding failures.

Repeated faces and pages make pair scores correlated; 75 pairs are not 75 independent identities. This is a small diagnostic sample, especially if difficult examples were deliberately added. Keep calibration and heldout sequences separate; compare a later untouched sequence sample before making general claims or tuning a threshold. The seven smoke-test labels remain separate from expanded analysis.

Analysis can run on a partial reviewed batch for diagnostics, but reach 50–100 confident labels across several sequences, with both classes and the requested hard cases, before deciding what to do next.''')
    code('''if BATCH_LABELED is not None:
    analysis_rows = review.read_csv(BATCH_LABELED)
    if SUPPLEMENT_LABELED is not None:
        analysis_rows += review.read_csv(SUPPLEMENT_LABELED)
    review.validate_rows(analysis_rows)
    ANALYSIS = SESSION / f'analysis_{datetime.now():%H%M%S_%f}'
    report = review.analyze_scores(analysis_rows, ANALYSIS)
    print(json.dumps(report, indent=2))
    display(Image(filename=str(ANALYSIS / 'same_vs_different.png')))
    print('Saved analysis:', ANALYSIS)
else:
    print('Import the expanded human labels before running score analysis.')
''')
    md('''## Outputs and rerunning

All new artifacts are under the printed `SESSION` directory: blind HTML reviews, candidate CSVs, imported labeled CSVs, optional expanded detection/embedding artifacts, and analysis files. Original smoke-run files, manual annotations, and the 0.8 threshold are preserved. No labels are inferred from model scores.

Notebook cells are deliberately unexecuted in the delivered file. Run top to bottom, pausing for browser review and setting the download paths before import. To resume an earlier session after restarting the kernel, set `BATCH_LABELED` to its saved labeled CSV (and `SUPPLEMENT_LABELED` if applicable), then run the analysis cell. Use a fresh `SESSION` for a new candidate review.

Regenerate this notebook after changing its authoring source with `python notebooks/build_pair_review_notebook.py` from the experiment directory. The generator only writes notebook 04; it never executes models or supplies labels.''')
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='AppliedTask (CUDA GPU)', language='python', name='venvappliedtask'),
        language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
    # Stable cell IDs make notebook diffs and regeneration reproducible.
    for index, cell in enumerate(cells):
        cell['id'] = f'pair-review-{index:02d}'
    path = Path(__file__).with_name('04_pair_review_and_score_analysis.ipynb')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
