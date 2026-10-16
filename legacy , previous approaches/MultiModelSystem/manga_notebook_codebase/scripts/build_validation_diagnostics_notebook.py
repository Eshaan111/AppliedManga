"""Generate the downstream validation diagnostics notebook (clears executions)."""
import json
from pathlib import Path
from build_text_extraction_notebook import cell

ROOT = Path(__file__).resolve().parents[1]

cells = [
    cell('markdown', '''# Validation diagnostics — step 2
Run **`../TextExtractionPipeline.ipynb` with `SPLIT = "val"` first**. This notebook evaluates its saved, filtered and joined `candidates.jsonl`.

It does not run OCR, rebuild datasets, filter text, or join fragments. Reference transcripts are used only for comparison. Input candidates and upstream output files remain unchanged.
'''),
    cell('code', '''from pathlib import Path
import json
import sys

ROOT = next((candidate for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             for candidate in (base, base / "manga_notebook_codebase")
             if (candidate / "src/text_extraction_diagnostics.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Start this notebook inside the project or workspace")
sys.path.insert(0, str(ROOT / "src"))
from text_extraction_diagnostics import load_inputs, run_diagnostics
from data_utils import load_jsonl
from ocr_reports import get_ocr_result

CANDIDATES_PATH = ROOT / "outputs/text_extraction/val/candidates.jsonl"
OUTPUT = ROOT / "outputs/diagnostics/ocr_joining_validation"
SAMPLE = 4   # index within the saved extraction results, zero-based
PAGE = 0     # page within its three-page sequence, zero-based
print("Upstream results:", CANDIDATES_PATH)
'''),
    cell('markdown', '''## 1. Load and verify saved extraction results
Check candidate/audit hashes, saved split, image hashes, and source raw OCR hashes. The evaluated sequences must belong to validation. A limited validation extraction is supported, and coverage is reported explicitly. If upstream results are absent or stale, fix/rerun the extraction notebook first.
'''),
    cell('code', '''rows, candidates, raw, audit, references, metadata, _, total = load_inputs(ROOT, CANDIDATES_PATH)
print(f"Evaluating {len(rows)} of {total} validation sequences / {len(rows) * 3} pages")
print("Saved final groups:", sum(len(p) for r in candidates.values() for p in r["pages"]))
print("Upstream filtering settings:", metadata["config"])
print("Sound-word blacklist:", ", ".join(metadata.get("sound_effect_words", [])))
'''),
    cell('markdown', '''## 2. Compare the exact saved groups with validation references
Matching is one-to-one fuzzy text comparison: threshold 0.5; high similarity 0.9. Similarity ignores punctuation and spacing. It does not establish correct balloon boundaries, speakers, or manga reading order.

Outputs: `allocations.jsonl`, `comparison.csv`, `summary.json`, HTML page reports, raw/final PNG previews, and readable transcripts. Group IDs, source IDs, boxes and text are taken from the pipeline output unchanged; references never replace extracted text. Diagnostics do not create a second `joined.jsonl`.
'''),
    cell('code', '''summary = run_diagnostics(ROOT, CANDIDATES_PATH, OUTPUT)
print(json.dumps(summary, indent=2))
print("Comparison CSV:", OUTPUT / "comparison.csv")
print("All-page report:", OUTPUT / "index.html")
'''),
    cell('markdown', '''## 3. Inspect a page: raw OCR, final groups, exclusions, and reference matches
Change `SAMPLE` and `PAGE` above. The left image shows the original raw candidates; the right shows the exact filtered and joined pipeline groups. Excluded candidates are listed below, with their recorded reasons.
'''),
    cell('code', '''import html
from IPython.display import display, HTML
from PIL import Image

if not 0 <= SAMPLE < len(rows) or not 0 <= PAGE < 3:
    raise ValueError("SAMPLE or PAGE is outside the saved results")
result = get_ocr_result(ROOT, sample=SAMPLE, page=PAGE, output=OUTPUT)
prefix = f"sample_{SAMPLE}_page_{PAGE}"
print(result["sequence_id"], "/ page", PAGE)
print("Original raw OCR boxes:")
with Image.open(OUTPUT / f"{prefix}_raw.png") as image:
    display(image.copy())
print("Final filtered and joined groups from TextExtractionPipeline:")
with Image.open(OUTPUT / f"{prefix}_groups.png") as image:
    display(image.copy())

columns = ["Group", "Source OCR IDs", "Extracted text", "Target index", "Reference text", "Similarity"]
values = [[g["id"], ", ".join(g["candidate_ids"]), g["text"], g["target_index"],
           g["target_text"], round(g["similarity"], 3) if g["similarity"] is not None else None]
          for g in result["groups"]]
values += [["Unmatched target", "", "", t["target_index"], t["text"], None]
           for t in result["unmatched_targets"]]
def escaped(value):
    return html.escape("" if value is None else str(value))
table = "<table><thead><tr>" + "".join(f"<th>{escaped(c)}</th>" for c in columns) + "</tr></thead><tbody>"
table += "".join("<tr>" + "".join(f"<td>{escaped(v)}</td>" for v in row) + "</tr>" for row in values)
display(HTML(table + "</tbody></table>"))
print("Upstream exclusions:")
print(json.dumps(result["filter_exclusions"], indent=2, ensure_ascii=False))
print("Saved preview:", result["preview_path"])
'''),
]


def main():
    for index, item in enumerate(cells):
        item['id'] = f'validation-diagnostics-{index:02}'
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3 (OCR environment)', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
    path = ROOT / 'notebooks/ocr_text_extraction/validation_ocr_joining.ipynb'
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
