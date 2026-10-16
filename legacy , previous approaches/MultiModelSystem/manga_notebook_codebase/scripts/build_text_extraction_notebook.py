"""Generate the maintained TextExtractionPipeline notebook (clears executions)."""
import json
import argparse
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell('markdown', '''# Text extraction pipeline
Load the saved sequence split → obtain raw OCR → filter non-English text, sounds, watermarks, and page numbers → join fragments → save `candidates.jsonl`.

Default: all 16 validation sequences, reusing the original raw cache. Set `SPLIT` to `train`, `val`, `test`, or `all`. For unseen images, set `INPUT_ROOT`; labels are never read. Missing raw OCR is extracted by the separate Python 3.11 OCR environment.

Filtering uses editable **whole-fragment** sound-word and watermark lists before joining. Watermarks include `MANGA`, `DO NOT MIRROR`, `STREAM.COM`, `DONOTMIRROR`, and `.COM`, including complete combinations. Matching ignores case, extra whitespace, and surrounding punctuation. Words inside dialogue sentences are preserved. Non-English fragments, SFX captions, repeated listed sounds, and small margin page numbers are removed before joining. Rerun the import cell after changing pipeline code. This heuristic can remove a listed word spoken by a character, and will miss unlisted or misrecognized sound effects.
'''),
    cell('code', '''from pathlib import Path
import json
import os
import sys

ROOT = None
for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    for candidate in (base, base / "manga_notebook_codebase"):
        if (candidate / "src/text_extraction_pipeline.py").is_file():
            ROOT = candidate
            break
    if ROOT is not None:
        break
if ROOT is None:
    raise RuntimeError("Start this notebook inside the project or workspace")
sys.path.insert(0, str(ROOT / "src"))
import importlib
import text_extraction_pipeline
importlib.reload(text_extraction_pipeline)
from text_extraction_pipeline import select_rows, run_pipeline, load_sound_words
from data_utils import load_jsonl
print("Project:", ROOT)
'''),
    cell('code', '''# Change these values, then rerun the following cells.
SPLIT = "val"                   # train / val / test / all
INPUT_ROOT = None               # e.g. Path("C:/data/new_manga"); images/<id>/01.png,02.png,03.png
LIMIT = None                    # None = entire selected split; 1 = one triplet
MIN_OCR_SCORE = 0.0             # Keep all valid confidence levels by default
MAX_JOIN_GAP = 1.25
CENTER_TOLERANCE = 1.5
SOUND_EFFECTS_PATH = ROOT / "datasets/rules/common_sound_effects.json"
RUN_NAME = SPLIT if INPUT_ROOT is None else "unseen"
OUTPUT_DIR = ROOT / "outputs/text_extraction" / RUN_NAME
RAW_CACHE = (ROOT / "outputs/ocr/paddleocr/candidates.jsonl"
             if INPUT_ROOT is None else OUTPUT_DIR / "raw_candidates.jsonl")
OCR_PYTHON = Path(os.environ.get("MANGA_OCR_PYTHON", str(ROOT.parent / ".venvOCR/Scripts/python.exe")))
'''),
    cell('markdown', '''## Load the split and inspect inputs
The saved split is loaded directly from `outputs/split.json`. Selection uses sequence IDs and ordered image paths; training targets and reference transcripts never enter inference. `INPUT_ROOT` bypasses the saved split for unseen image folders.
'''),
    cell('code', '''source, selected_rows = select_rows(ROOT, SPLIT, INPUT_ROOT, LIMIT)
print(f"Selected {len(selected_rows)} sequences / {len(selected_rows) * 3} pages")
print("First sequence:", selected_rows[0]["sequence_id"])
print("Images:", [str(p) for p in selected_rows[0]["images"]])
sound_words = load_sound_words(SOUND_EFFECTS_PATH)
print(f"Sound-effect list ({len(sound_words)} words):", ", ".join(sorted(sound_words)))
print("Final output:", OUTPUT_DIR / "candidates.jsonl")
'''),
    cell('markdown', '''## Run extraction, filtering, and joining
Existing raw candidates remain unchanged. Missing sequences are extracted into this run's `raw_candidates.jsonl`. Sound words are removed **before** geometry-based joining, so they cannot become part of neighboring dialogue groups.

Final candidates contain `id`, `text`, pixel `box`, mean OCR `score`, and page-local source `candidate_ids`. The three page lists remain in image order. Group ordering is spatial; this stage does not solve manga panel reading order or assign speakers.
'''),
    cell('code', '''records, summary = run_pipeline(
    ROOT, split=SPLIT, input_root=INPUT_ROOT, raw_cache=RAW_CACHE,
    output_dir=OUTPUT_DIR, ocr_python=OCR_PYTHON, limit=LIMIT,
    min_score=MIN_OCR_SCORE, max_gap=MAX_JOIN_GAP,
    center_tolerance=CENTER_TOLERANCE, sound_effects_path=SOUND_EFFECTS_PATH,
)
'''),
    cell('markdown', '''## Verify and inspect the saved output
`filter_audit.jsonl` records every discarded fragment and its reason. `candidates.metadata.json` records the split, settings, blacklist, code, image and input/output hashes. Reused legacy OCR is explicitly marked as lacking its original runtime/image provenance; regenerate raw OCR into a fresh cache for a fully fingerprinted run.
'''),
    cell('code', '''final_path = OUTPUT_DIR / "candidates.jsonl"
saved = load_jsonl(final_path)
assert [r["sequence_id"] for r in saved] == [r["sequence_id"] for r in selected_rows]
assert all(len(r["pages"]) == 3 for r in saved)
for record in saved:
    for page in record["pages"]:
        source_ids = [cid for group in page for cid in group["candidate_ids"]]
        assert len(source_ids) == len(set(source_ids)), "A fragment was joined more than once"
audit = load_jsonl(OUTPUT_DIR / "filter_audit.jsonl")
sound_removals = [dict(sequence_id=p["sequence_id"], page_index=p["page_index"],
                      text=e["candidate"]["text"])
                  for p in audit for e in p["excluded"] if e["reason"] == "sound_effect_word"]
print("Removed sound-effect fragments:", len(sound_removals))
print(json.dumps(sound_removals[:10], indent=2, ensure_ascii=False))
print("First page's joined candidates:")
print(json.dumps(saved[0]["pages"][0], indent=2, ensure_ascii=False))
print("Saved:", final_path)
'''),
    cell('markdown', '''## Next step: validation diagnostics
After a validation run, open [validation_ocr_joining.ipynb](ocr_text_extraction/validation_ocr_joining.ipynb).
It evaluates this notebook's saved `outputs/text_extraction/val/candidates.jsonl`, generates reference comparisons and page previews, and lists the upstream filtering exclusions. It does not rerun OCR, filtering, or joining.
'''),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Execute generated Python cells in this interpreter and save their outputs')
    args = parser.parse_args()
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3 (OCR environment)', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')),
        nbformat=4, nbformat_minor=5)
    for index, item in enumerate(notebook['cells']):
        item['id'] = f'text-extraction-{index:02}'
    path = ROOT / 'notebooks/TextExtractionPipeline.ipynb'
    if args.execute:
        namespace, count = {}, 0
        for item in notebook['cells']:
            if item['cell_type'] != 'code':
                continue
            count += 1
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(''.join(item['source']), str(path), 'exec'), namespace)
            item['execution_count'] = count
            if stream.getvalue():
                item['outputs'] = [dict(output_type='stream', name='stdout',
                                        text=stream.getvalue().splitlines(keepends=True))]
        notebook['metadata']['execution'] = dict(interpreter=sys.executable, python=sys.version,
                                                 mode='Python cells executed sequentially without a Jupyter kernel')
        print(json.dumps(namespace['summary'], indent=2))
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
