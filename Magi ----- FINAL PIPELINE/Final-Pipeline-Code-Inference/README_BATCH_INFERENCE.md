# Batch sequence inference (.py)

Entry point: [run_sequence_inference.py](run_sequence_inference.py). This processes the selected split from `Magi/TaskData/sequences.json`, using its sequence IDs and ordered image paths. It never reads reference labels for inference and does not run scoring. The notebook version includes the interactive viewers.

## Run

PowerShell, from the Magi directory:

```powershell
# All 15 current test sequences.
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe paddleOCR-Magi/run_sequence_inference.py --split test

# All 80 current development sequences.
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe paddleOCR-Magi/run_sequence_inference.py --split development
```

The number of sequences is taken from your manifest rather than hardcoded. Use your working CUDA environment and the same local resources as the [full pipeline](README_ARTIFACT_CONTOUR_FULL_INFERENCE.md). No dependencies are installed or models downloaded by this script.

Useful first checks:

```powershell
# Verify split selection and all image paths; no inference or output changes.
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe paddleOCR-Magi/run_sequence_inference.py --split development --dry-run

# Run only the first sequence before committing to the full split.
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe paddleOCR-Magi/run_sequence_inference.py --split development --limit 1 --output-dir submission_outputs/first_sequence
```

Use `--sequence-ids seq_one seq_two` for selected IDs, `--task-data C:/path/TaskData` for another dataset, and `--model-config paddleOCR-Magi/batch_models.example.json` to override local model/checkpoint paths. The example JSON has the current paths. Relative override paths resolve from your command's working directory.

`--no-cleanup` skips only final Gemma text correction; Gemma artifact classification still runs. `--force` regenerates this runner's own stage caches. `--overwrite` permits replacing an existing combined prediction file, but publication still occurs only after all selected sequences succeed. With unchanged settings, rerunning uses compatible own-stage caches rather than blindly trusting an old final transcript.

## Files produced

Default final paths, relative to Magi:

```text
submission_outputs/inference_test/predictions.jsonl
submission_outputs/inference_development/predictions.jsonl
```

Each contains one object per sequence, in manifest order:

```json
{"sequence_id":"seq_example","pages":[[{"speaker":"character1","text":"HELLO!"}],[],[]]}
```

There are exactly three ordered page lists and exactly speaker/text fields per entry. Names are sequence-local; NARRATION/UNKNOWN retain the pipeline's current meanings. The actual `TaskData/score.py` schema loader validates every per-sequence output and the combined file.

The selected output directory also contains:

- `batch_status.json`: requested/completed sequence IDs, failure status and runtime/helper hashes.
- `sequences/<sequence_id>/transcript.jsonl`: final per-sequence transcript.
- `sequences/<sequence_id>/transcript_raw.jsonl`: transcript before final cleanup.
- `sequences/<sequence_id>/inference.log`: model/stage progress and tracebacks.
- `sequences/<sequence_id>/inference_config.json`: resources and image inputs.
- The same artifact, contour, grouping, character and transcript audit exports as the notebook.

Default own stage caches are at `paddleOCR-Magi/cache_batch_artifact_contour/<sequence_id>/<image_path_scope>/`. Older experiment and notebook cache folders are not searched. Use `--cache-root` to choose a different dedicated location. Final prediction JSONL and score reports under submission_outputs remain uploadable under the root .gitignore; feature caches and logs are excluded.

A failed sequence/stage stops the batch, records the phase and sequence in stage_status.json, and leaves any previous combined prediction file untouched. No empty placeholder transcript is invented. Per-sequence logs/exports and stage caches remain available for diagnosis and reruns.

## How the runner works

[The worker](src/artifact_contour_sequence.py) contains readable plain Python extracted from the current notebook's inference cells, including the latest numeric-preservation cleanup examples. It imports the same contour helpers from the parent folder. It does not read/execute the notebook at runtime. Original function bodies and prediction settings are preserved; input/resource/output plumbing is configurable, tables/viewers are omitted, and a shared-stage scheduler reuses each model across sequences while rebuilding sequence-specific state.

The runner uses one batch process, with model work scheduled by stage. It does not load models for every sequence. Prior per-sequence behavior is preserved under LegacyMagiPipelines for regression comparison. A fresh namespace/image set is created per sequence per phase, so entities, OCR context and reading order do not cross sequences. Prior completed stages are read from validated caches; if a prior stage is missing/stale during a later phase, the batch stops rather than silently rerunning it with the wrong model state. The worker is a snapshot; future changes to notebook inference/prompt code must also be applied to this worker. Its header records the source notebook hash, and each batch records the worker/helper hashes.

## Score later

For a complete development export, run the official evaluator separately from Magi:

```powershell
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe TaskData/score.py --references TaskData/development/labels.jsonl --predictions submission_outputs/inference_development/predictions.jsonl --output submission_outputs/inference_development/score_results.json
```

Test labels are not currently provided, so test predictions can be exported but no test accuracy report can be calculated yet. For a subset/limit run, supply a reference file containing exactly that subset when scoring; the full development reference otherwise penalizes unprocessed sequences as missing. Never overwrite labels.jsonl with predictions.

## Verification performed

Python syntax checks; exact AST comparison of all 34 copied inference functions against the notebook; a dry-run selection of all 15 test sequences; and two-sequence batch tests with a stub worker using the official schema loader. Tests also rejected duplicate sequence IDs, missing images and incorrect page counts, and confirmed a failed run preserves a previous combined export. These are implementation checks, not a real-model inference comparison or an accuracy benchmark.


## Shared-model schedule and workflow tests

The order for every sequence is unchanged. Only batch scheduling/model lifetime changed:

1. Original Magi detections, OCR and links for all sequences.
2. Fine-tuned character embeddings for all sequences, using the same Magi instance. All original predictions finish before the first crop-weight patch. Then unload Magi.
3. Paddle OCR for all sequences, then unload Paddle.
4. Gemma artifact classification for all sequences, then unload Gemma.
5. Existing character cleanup/entities/text mapping and YOLO bubble detection, then unload YOLO.
6. Existing punctuation bubble gate, contour grouping/roles, tail override, speaker routing and raw exports.
7. Gemma final OCR correction for all sequences, then unload Gemma.

Models load lazily only when a stage cache needs inference. On an uncached batch, Magi loads once across steps 1-2; Paddle and YOLO once each; Gemma once for artifacts and once for cleanup. The CPU crop checkpoint is loaded once per batch. Earlier preprocessing/metadata/cache validation can repeat per phase; the model resource pool does not eliminate all CPU or disk work. No-cleanup runs skip final correction, and fully valid cached model stages do not load those models.

`--force` refreshes only the current phase's stages. It does not rerun original Magi after crop weights are patched. Model prompts, generation settings, thresholds, grouping rules, tail policy and speaker routing remain unchanged. Both [src/artifact_contour_batch.py](src/artifact_contour_batch.py) and [src/artifact_contour_sequence.py](src/artifact_contour_sequence.py) are runtime dependencies.

Run the CPU-only regression tests from Magi:

```powershell
& C:/Users/ESHAAN/BYTE/godfather.venv/Scripts/python.exe paddleOCR-Magi/pipeline_review/test_batch_workflow.py
```

Five tests passed. They verify unchanged prediction-function ASTs after removing only loading wrappers, identical prompts/thresholds and cache validation, saved-example transcript/member-speaker routing parity, original-detection-before-patching order, model factory counts, sequential-versus-staged fixture output equality, cache-only reruns, force behavior, missing-prior-stage rejection, and preservation of an existing combined export after failure.

Tests use stubs and saved data. No real GPU inference or real-model numerical parity run has been performed, so they cannot guarantee bitwise equality of GPU outputs or rule out every third-party model-state issue. Per-sequence logs remain under sequences/<sequence_id>/inference.log; stage_status.json reports shared-stage progress/failure.

## Current folder names

The code now lives under `Magi ----- FINAL PIPELINE/Final-Pipeline-Code-Inference/`. The earlier commands and some runtime defaults above use the old folder names from the saved run. Update those paths before rerunning; see the [folder-name note](README_ARTIFACT_CONTOUR_FULL_INFERENCE.md#folder-name-note). The root [requirements.txt](../../requirements.txt) records the current local dependencies.
