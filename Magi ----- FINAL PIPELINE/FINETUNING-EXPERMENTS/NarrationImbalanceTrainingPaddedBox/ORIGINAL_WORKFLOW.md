# NarrationImbalanceTraining

This experiment is written for our local Magi project:

`C:\Users\ESHAAN\BYTE\Applied-Task\Magi\FINETUNING-EXPERMENTS\NarrationImbalanceTraining`

The complete agreed workflow is included in [WORKFLOW.md](WORKFLOW.md). The implementation uses plain functions and loops. `config.py` holds paths/settings and the small cache helpers; `model.py` holds the model, padding and shared metrics. Preparation and training code stays in the notebooks.

## Start here

Open this folder in Jupyter and select **Godfather (Python 3.11 | Torch + Paddle GPU)**. This existing kernel uses `C:\Users\ESHAAN\BYTE\godfather.venv\Scripts\python.exe`; PyTorch imports and CUDA is available. Notebook metadata already selects it. If a dependency is missing, install `requirements.txt` into that kernel. Keep its existing CUDA PyTorch build. The local Magi code uses Transformers 4.52.4.

Edit `config.py` if needed, then run **00 → 01 → 02 → 03 → 04 → 05 → 06**. Restart the kernel after changing config. The defaults already point at the existing project files.

| Notebook | What it does / cache created |
|---|---|
| 00_config_and_inspect | Prints kernel, dependencies and paths; writes `cache/discovery.json` |
| 01_prepare_sequence_cache | Reuses raw Magi/OCR, matches GT, keeps all boxes; writes `cache/sequences.json` |
| 02_cache_deberta_features | Frozen previous-two-box semantic features; writes `cache/deberta.pt` |
| 03_cache_magi_features | Frozen Magi text-object vectors and tail affinities; writes `cache/magi.pt` and reusable `cache/magi_pages/*.pt` |
| 04_build_training_samples | Joins modalities, groups overlapping sequences, splits, computes weights; writes `cache/samples.pt` |
| 05_train_classifier | Oversamples intact training sequences; writes checkpoint, history and validation predictions |
| 06_evaluate | Loads best checkpoint, exports metrics/predictions/errors |

Each notebook begins with configuration and cache discovery/reuse. Its following cells create its output if missing/stale. Missing prerequisite data caches call the relevant earlier notebook, printing which one runs. Stages 04/05 also check upstream feature fingerprints. Stage 06 requires an existing trained checkpoint. Running stage 05's training cells starts a fresh run.

## Existing files we reuse

- GT: `Magi/TaskData/development/labels.jsonl`.
- Images: `Magi/TaskData/development/images/<sequence_id>/01.png`, `02.png`, `03.png`.
- Primary raw OCR/detections: `Magi/paddleOCR-Magi/cache_baseline/development/<sequence_id>/magi_raw.json` — 80 sequence caches found.
- Fallback raw cache: `Magi/FINETUNING-EXPERMENTS/TextRoleVL/cache/development`.
- Older visual vectors: `Magi/FINETUNING-EXPERMENTS/isNarratorTextMagi/narration_data.pt` — 1,490 labeled-box samples, 256-dimensional DETR vectors.
- Magi model: `C:\MODELS\magiv2`.
- Semantic model: `C:\MODELS\deberta-v3-base`. Both tokenizer and model use `local_files_only=True`; missing files cause an error instead of a download.

The raw cache's `data` is three page dictionaries. A page's `texts` contains original-pixel xyxy boxes; `ocr` contains strings in the same order. `settings.pages` supplies image names/SHA256s; `data_hash` checks the payload. This is the actual local format, not an invented interchangeable schema. Fresh raw cache creation uses frozen Magi and OCR only when neither existing source has that sequence.

GT rows contain `sequence_id` and three page lists of `{speaker, text}`. Page-local normalized OCR matching follows `isNarratorTextMagi/createNarrationData.ipynb`: similarity >=95 and margin >=5. If several boxes claim one GT line, all those labels are ignored. Inspect the counts printed by stage 01: matching quality limits supervision.

`characterN → 0 / SPOKEN`, `NARRATION → 1 / NARRATED`; unknown, unmatched, ambiguous and other labels → `-100`. Every real detected box remains available as context. `is_essential_text` is never read or used.

## Feature and cache contracts

`sequences.json` contains a list of `{sequence_id, pages, boxes, raw_cache}`. Pages include path, SHA256 and dimensions. Boxes include ID (`PG1:T1`), zero-based page/text indices, OCR, pixel bbox, GT speaker and label. Page order is sorted filename order; existing within-page box order is preserved.

`deberta.pt` contains `{signature, vectors}`; `vectors[sequence_id][box_id]` is float `[768]`. Inputs concatenate previous two same-page OCR strings and target, separated by SEP tokens. There is no previous-page text or future OCR in this encoder input. Default length is 256 tokens. Older context is discarded first; an overlong target keeps its last tokens. Empty target OCR uses UNK. Mean-pool target token states only. Encoder padding uses its own attention mask. DeBERTa runs frozen in eval/inference mode.

`magi.pt` contains `{signature, vectors, tails, legacy_vectors_reused}`. Vectors use the same two IDs and are float `[D]` (our Magi D=256). Tails are float `[2]`: maximum sigmoid text–tail affinity and whether it exceeds 0.3. Candidate tails pass detection threshold 0.34; no detected tails gives `[0,0]`.

The old vector cache includes only labeled boxes; raw JSON stores discrete tail associations, not continuous scores. Stage 03 must therefore run frozen Magi once for missing complete features, including unlabeled boxes. It reuses old vectors when page/index/bbox and numerical vector comparison agree. One-to-one Hungarian IoU >=0.7 aligns original cached boxes with the full DETR object query set, without repeating text score/NMS filtering; failure stops the notebook rather than dropping boxes or filling fake vectors. Each extracted page is saved immediately, so restarting stage 03 reuses completed pages. Magi loads only when a page cache is missing.

`samples.pt` contains CPU tensors per full sequence: `text[N,768]`, `visual[N,D]`, `geometry[N,4]`, `tail[N,2]`, `page[N]`, `labels[N]`, plus original box/page metadata, train/validation indices, train counts, class weights and maximum token count. Geometry is `[xc/W, yc/H, width/W, height/H]`; it is not clipped because local detector boxes can extend slightly beyond page edges.

Feature signatures include sequence data, settings and local model file sizes/modification times. Reuse canonical files only with matching signatures. After editing GT/raw caches or OCR matching settings, remove `cache/sequences.json` and rerun stages 01–06. Stages 02–05 refresh dependent feature/sample caches when fingerprints differ. Delete feature caches explicitly if replacing model weights without changing file metadata. Keep model.py unchanged when loading a trained checkpoint; retrain after architecture edits. Local legacy `.pt` uses `weights_only=False`; canonical tensor caches use `weights_only=True`.

## Split, sampler, model and masks

A sample is one complete three-page sequence. Sequences sharing an image hash are kept together. The seeded split search targets 20% validation with both classes in both splits. Validation is used for checkpoint selection, not an independent test set.

The training sampler makes `len(train)` draws with replacement per epoch. Narration-containing sequences have sampling weight 2 and others 1. No box is duplicated within a sequence. Sequences with no labels remain cached but are omitted from training because they supply no supervised signal; validation retains them.

Text 768→256, visual D→256, geometry 4→32, tail 2→16 are ReLU projections. Concatenate 560 features, project to 256, then add learned sequence-position and page embeddings. A two-layer TransformerEncoder uses 4 heads, feed-forward width 512, dropout 0.1; Linear(256,2) classifies every box. It attends across the complete sequence without a causal mask. Encoders stay frozen through cached features.

| Kind | attention_mask | label | loss |
|---|---:|---:|---|
| Real labeled box | True | 0 or 1 | yes |
| Real unlabeled box | True | -100 | no |
| Padding | False | -100 | no |

PyTorch's `src_key_padding_mask=True` means ignore, so the model passes `~attention_mask`.

## Class-balanced loss and logs

Count each labeled box once in the original training split, before oversampling. For default beta=0.999, compute `raw_w[c]=(1-beta)/(1-beta**n[c])`, then normalize weights to mean 1. Zero-support classes in a split are rejected. Beta=0 gives equal weights.

For labeled boxes, `L_i=-w[y_i] log P(y_i)`. Batch loss is `sum(L_i)/number_of_labeled_boxes`; epoch loss uses the same sum/count across batches. For batch size 1 this is the workflow's sequence loss. This differs from weighted CE's built-in mean, which divides by the sum of weights. Ignored and padded labels contribute nothing. There is no additional per-sequence class balancing or focal loss.

`runs/history.json` records train/val loss, per-class precision/recall/F1/support, macro F1, balanced accuracy, and per-class true-label probability mean/median, fraction >0.9 and fraction <0.5. Mean batch L2 gradient norms for fusion, Transformer and classifier are logged before clipping. Training metrics reflect the sampled sequences; validation uses each sequence once.

`runs/best.pt` is selected by validation macro F1. Per-epoch and best validation JSONs contain every real box, including ignored boxes. Stage 06 writes `evaluation_metrics.json`, `evaluation_predictions.json`, `validation_predictions.csv` and `validation_errors.csv`: sequence/box ID, OCR, GT, prediction, probabilities, tail evidence, page and bbox. Ignored boxes are excluded from metrics/errors. These diagnostics support a later focal-loss decision; no automatic unfreezing is included.

See [VALIDATION.md](VALIDATION.md) for checks actually performed. The delivered setup contains no trained experiment or generated model features; you run those notebooks in your kernel.

Stage 03 alignment fix: old filtered page caches remain reusable when they cover all original boxes. Failed pages are refreshed using all object queries. The stage signature changed so downstream samples refresh automatically.


## 07 — Five training seeds, then 08 — Optuna tuning

Run `07_train_five_seeds.ipynb` first, then `08_optuna_tuning.ipynb`, using the existing Godfather kernel. These reuse the exact `cache/samples.pt` split and cached class weights. They do not change labels or regenerate an existing sample cache. The original baseline checkpoint is checked against the cached samples, and its saved training settings are reused.

07 uses seeds `[42, 123, 2026, 7, 99]`, fixed dropout rate 0.1, and the same original architecture/optimizer settings/epochs. Seeds vary initialization, stochastic dropout masks and sequence sampling. Each seed saves its best macro-F1 epoch, full history, checkpoint and per-box prediction CSV in `runs/five_seeds/seed_<seed>`. Aggregate `per_seed_metrics.csv` and `seed_summary.csv` report narration precision/recall/F1 and macro F1: mean, sample std (ddof=1), min, max and range. Completed matching runs are reused; interrupted seeds restart. Use a new RUN_NAME after changing settings.

08 needs Optuna in your kernel: run `%pip install "optuna>=4,<6"` once. It defaults to 20 total trials (not 20 additional trials on every rerun), each evaluated across three fixed seeds `[42,123,2026]`. It searches learning rate 1e-5–1e-3, weight decay 1e-4–0.1, dropout 0.1/0.2/0.3 and narration sequence weight 1/2/3. The objective is mean macro F1 across the three independently trained seeds, with each checkpoint selected by macro F1. Class-balanced weights, architecture, batch size, epochs and decision threshold are fixed. The original configuration is queued first; trials run sequentially with seeded TPE and no pruning. SQLite stores the study under `runs/optuna_tuning`; trial CSVs, per-seed histories/checkpoints and best parameters are saved alongside it.

The chosen configuration is reported across all five baseline seeds, reusing its completed search-seed runs. The notebook saves a baseline-vs-tuned aggregate CSV and verifies matching baseline split/seeds/settings. Same-validation tuning scores remain development results, not independent test estimates. Resume preserves trial records, but future TPE suggestions may differ from an uninterrupted run. Study manifests reject data/code/search-setting changes; choose a new OUTPUT folder for another study.

`training_runs.py` is the shared plain training loop. `BoxClassifier` now accepts an optional dropout rate; its default remains 0.1 and old checkpoints retain the same parameter shapes and default behavior. `runs/best.pt`, existing prediction files and feature caches are not overwritten by these notebooks.
