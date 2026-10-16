# OCR and dataset foundation

The project currently retains OCR extraction, geometry-based line joining,
visual reports, task data, and reproducible training/validation datasets.
The objective is a reproducible system that works on unseen images. Every
processing step is implemented in code. Reference labels are for training
and evaluation; they never supply or correct inference text.

## Structure

- `TaskData/`: original images, manifest, labels, sample submission, official scorer.
- `datasets/`: preserved train/validation/test manifests, references, component
  tasks, and annotation templates for the future larger system.
- `src/`: OCR, joining, reports, dataset preparation, and evaluation helpers.
- `scripts/`: runnable extraction, joining, dataset preparation, and comparison tools.
- `notebooks/`: data inspection/preparation and the two OCR joining workflows.
- `outputs/ocr/paddleocr/candidates.jsonl`: original cache, preserved unchanged
  (16 sequences, 48 pages, 1,456 raw text fragments).
- `outputs/split.json`: original sequence-level 64/16 development split.
- `outputs/diagnostics/`: retained OCR diagnostics and joining reports.

Qwen training/inference, identity experiments, adapters, and their outputs have
been removed. `scripts/cleanup_legacy.ps1` records the explicit cleanup paths;
it defaults to a dry run.

## OCR runtime

From the workspace root, use the retained Python 3.11 `.venvOCR` environment.
For a fresh environment, install `requirements-ocr-lock.txt` to reproduce the
recorded full package set. `requirements-ocr.txt` pins the direct dependencies.
Optional notebooks additionally use `jupyter`, `nbformat`, and `matplotlib`
from `requirements.txt`; the extraction CLI does not need them.

OCR uses local PaddleOCR `PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` models,
CPU execution, disabled MKL-DNN, and disabled document/line orientation and
unwarping. The model folders already exist on this machine under
`C:/Users/ESHAAN/.paddlex/official_models/`. Each must contain `inference.json`,
`inference.pdiparams`, and `inference.yml`. On another machine, obtain these
models through [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and pass
`--model-root` pointing to their parent directory. The extraction command
requires local files and does not use hosted inference.

Validate all supplied images without extracting or writing anything:

```powershell
& '.venvOCR/Scripts/python.exe' manga_notebook_codebase/scripts/build_ocr_cache.py --inspect
```

Extract supplied test images into a fresh output:

```powershell
& '.venvOCR/Scripts/python.exe' manga_notebook_codebase/scripts/build_ocr_cache.py --split test --output manga_notebook_codebase/outputs/ocr/test/candidates.jsonl
```

For unseen data, supply `images/<sequence_id>/01.png, 02.png, 03.png` beneath
an input folder. Labels are unnecessary. Filenames determine page order unless
`sequences.json` supplies explicit ordered paths in the task manifest format.

```powershell
& '.venvOCR/Scripts/python.exe' manga_notebook_codebase/scripts/build_ocr_cache.py --input 'C:/data/new_manga' --output manga_notebook_codebase/outputs/ocr/new_manga/candidates.jsonl
```

Use `--limit 1` for one triplet, or repeat `--sequence-id ID` to select sequences.
New runs produce `candidates.metadata.json` with image/model/code hashes,
package versions, runtime settings, and output hash. Identical inputs and
runtime can resume; changes require a fresh output. Candidates contain
`sequence_id` and three page lists of `{id, text, score, box}` records.
Boxes use image-pixel `[x0, y0, x1, y1]` coordinates; IDs are page-local.

The original cache predates provenance recording. It remains readable by the
joining workflows, but extraction refuses to append to it or claim verified
provenance. Regenerate into a fresh location when needed. Reproducibility
records do not guarantee identical floating-point scores across machines.

## Joining without labels

```powershell
& '.venvOCR/Scripts/python.exe' manga_notebook_codebase/scripts/join_ocr_cache.py --candidates manga_notebook_codebase/outputs/ocr/new_manga/candidates.jsonl --output manga_notebook_codebase/outputs/ocr/new_manga/joined.jsonl
```

This runs the existing geometry-based joiner, preserves source candidate IDs,
and records input/output/code hashes and joining parameters. It reads no labels.
It proposes text groups and spatial order; panel reading order, speaker assignment,
and cross-page identity remain future stages. Raw OCR includes non-story text.

## Existing joining and dataset workflows

### TextExtractionPipeline

Open `notebooks/TextExtractionPipeline.ipynb` and run all cells. It loads the saved
split (`val` by default), reuses/extracts raw OCR, removes whole fragments matching
the common sound-effect words and watermark phrases in `datasets/rules/common_sound_effects.json`,
then joins the remaining fragments. Matching ignores case and surrounding
punctuation; it does not delete words inside sentences. Watermark exclusions
include `MANGA`, `DO NOT MIRROR`, `STREAM.COM`, `DONOTMIRROR`, and `.COM`, including
complete combinations such as `MANGA DO NOT MIRROR`. Exclusions run before joining
and watermark removals are audited as `watermark_phrase`. This word-list heuristic
can remove a listed word when spoken and misses unlisted visual sounds.

The final joined, filtered file is `outputs/text_extraction/val/candidates.jsonl`.
An audit records every removed fragment; metadata records source, image, code,
blacklist and output hashes. Change `SPLIT` or set `INPUT_ROOT` for unseen data.
No reference text or manual reviews enter this pipeline. The original raw cache
is preserved. To regenerate and execute the notebook's Python cells without
Jupyter, run `scripts/build_text_extraction_notebook.py --execute` using `.venvOCR`.
This command clears notebook edits and regenerates from the checked-in generator.

The following notebooks and their dependencies are preserved:

- `notebooks/00_inspect_data.ipynb`
- `notebooks/02_prepare_training_data.ipynb`
- `notebooks/ocr_text_extraction/validation_ocr_joining.ipynb`

`legacy/09_directory_joined_ocr.ipynb` is the older combined directory joining
and comparison workflow. It has been moved out of active notebooks because
extraction and diagnostics now run as separate steps. Data inspection and
training-data preparation notebooks remain available for the larger system.

Prepare datasets with `python manga_notebook_codebase/scripts/prepare_datasets.py`.
The stored sequence split is reused, keeping all three pages together.
After running `TextExtractionPipeline.ipynb` with `SPLIT="val"`, open
`notebooks/ocr_text_extraction/validation_ocr_joining.ipynb` for diagnostics.
It consumes the saved filtered/joined `outputs/text_extraction/val/candidates.jsonl`
and its audit/provenance, checks that inputs have not changed, and generates
reference comparisons and raw/final image previews. It never reruns OCR,
filtering, or joining, and does not rebuild datasets or modify extraction results.
The equivalent command is:

```powershell
& '.venvOCR/Scripts/python.exe' manga_notebook_codebase/scripts/join_validation_ocr.py
```

`join_directory_ocr.py` and `preview_ocr_joining.py` retain the existing visual
comparison workflows. Reference matching is diagnostic and never changes joined
text. Explicit manual exclusion reviews are optional human diagnostics; do not
use them as an unseen-data inference stage. Automated filtering is future work.

Run the retained regression tests with:

```powershell
& '.venvOCR/Scripts/python.exe' -m unittest discover -s manga_notebook_codebase/tests -v
```
