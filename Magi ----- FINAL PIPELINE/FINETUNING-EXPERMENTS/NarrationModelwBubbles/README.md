# NarrationModelwBubbles

Select **Godfather (Python 3.11 | Torch + Paddle GPU)** and run cells in order.

## Pipeline

1. Magi detects text, characters and associations and produces OCR.
2. Paddle reads the Magi text crops. The existing Unicode validity checks choose the OCR output, preferring Paddle when both pass.
3. YOLO detects bubbles. Keep a candidate only when it contains an assigned nonempty selected OCR box (at least 80% text-box coverage). OCR misses can still reject a real text bubble.
4. Save retained RGB crops. Grayscale-to-RGB crops pass through frozen Magi DETR; mean encoder tokens produce 256-dimensional visual features. Texts sharing a bubble share its vector.
5. Uncovered text retains its original Magi text-query vector, so narration outside bubbles remains present. Keep original full-page geometry and tail evidence.
6. Frozen DeBERTa embeds selected OCR with the previous two texts from the same page. The narration classifier combines text, visual, geometry, tail and sequence context.
7. The full inference notebook retains character cleanup, entity clustering, text-to-character association, speaker routing, preview and exports from the standard full pipeline.

No DINO features are used. Original labels and the baseline train/validation sequence split are preserved.

## Notebooks — one line each

- `NarrationModelwBubbles_Test.ipynb`: complete one-page pipeline; change `PAGE_NUMBER`, model path and checkpoint path in the settings cell.
- `NarrationModelwBubbles_FullPipeline.ipynb`: same complete pipeline; set `PAGE_NUMBER=None` to process the full test sequence.
- `NarrationModelwBubbles_Training.ipynb`: prepare/cache all development inputs, then choose training notebook 05 or 07.
- `01_prepare_sequence_cache.ipynb`: preserve original labels/boxes and cache Magi + Paddle selected OCR.
- `02_cache_deberta_features.ipynb`: cache selected-text DeBERTa embeddings.
- `03_cache_magi_features.ipynb`: cache retained bubble crops and DETR embeddings with original-query fallback.
- `04_build_training_samples.ipynb`: join features and reuse the baseline split.
- `05_train_classifier.ipynb`: train or resume the classifier using frozen cached text/bubble features.
- `07_finetune_deberta_and_classifier.ipynb`: independently select DeBERTa/classifier training, resume, validate and export the best pair.
- `06_evaluate.ipynb`: evaluate the new checkpoint on the held-out validation sequences.
- `BubbleDetector_executed_smoke.ipynb`: standalone detector copy with explicit local paths and cleared outputs; no models.json needed.
- `NarrationModelwBubbles_Test_executed.ipynb`: historical Magi-only outputs, retained as a legacy record.

## Running

Start with the test notebook to inspect crops and complete page outputs. Model/checkpoint paths are independently editable. It automatically uses `runs_selected_ocr/best.pt` after training; until then it uses the existing standard checkpoint for **exploratory predictions**. That checkpoint was not trained on bubble features.

Prepare inputs in the wrapper notebook, then use 05 or 07. Their first cell sets `TrainClassifier`, `TrainDeberta`, `UseClassifierCheckpoint`, checkpoint paths, `ResumeTraining`, `RUN` and total `EPOCHS`. In 05 DeBERTa must stay frozen; its selected encoder must match the semantic cache.

`ResumeTraining=True` loads RUN/last.pt when present, including model, optimizer/scaler, sampler/random states, epoch, history and best score. Initialization paths apply only to a new run. Change EPOCHS from 40 to 60 and rerun the training cell for epochs 41–60, or run all cells after a kernel restart. Both switches False evaluates only. Changing the trainable layers, train switches or data requires a new RUN.

Every epoch updates last.pt; only improved validation macro F1 updates best.pt. Epoch 0 is validated and protected as a candidate best. No automatic early stopping is used. Old runs without last.pt can initialize weights but cannot fully resume an optimizer. Both notebooks export the best classifier at RUN/best.pt. Notebook 07 also writes the best Hugging Face encoder to RUN/deberta/. RUN/inference_config.py selects the corresponding encoder. No epoch export folder is used.

## Caches

Current caches use `cache_selected_ocr/`; checkpoints and evaluation use `runs_selected_ocr/`. Selected OCR, crops, previews, per-page DETR features, DeBERTa features, joined samples and full pipeline outputs are cached separately. Fingerprints track pages, text/boxes, settings, model identities and helper code. Completed pages resume from compatible caches.

The earlier `cache/` and `runs/` folders are preserved as legacy Magi-only artifacts. They are not the corrected pipeline's ready-to-use caches. Selected-OCR caches and earlier training results are preserved. The redesigned resume notebooks have not been executed by Codex; run them in Godfather.

`prepare_cache.py --split both` prepares the current three-page test sequence and development data without training. `validate_cache.py` checks joins, preserved labels, split isolation and crop hashes after preparation.

## Independent checkpoint inputs

Both training notebooks expose `DEBERTA_BASE_PATH`, `DEBERTA_CHECKPOINT_PATH`, `DEBERTA_LAYER_CHECKPOINTS`, `CLASSIFIER_CHECKPOINT_PATH` and `CLASSIFIER_LAYER_CHECKPOINTS` in their first code cell. DeBERTa accepts a Hugging Face directory, joint best.pt/last.pt, or raw full encoder state dict. The base path supplies config/tokenizer when loading .pt weights. Classifier input accepts joint checkpoints, classifier checkpoints or raw full classifier state dicts. Raw classifier files use the architecture settings from config.py.

Layer dictionaries override the selected full model for a new run, for example `{11: Path('.../deberta_layers/layer_11.pt')}` or `{'transformer.layers.1': Path('.../classifier_layers/transformer.layers.1.pt')}`. Partial loading is strict; incompatible dimensions raise an error. ResumeTraining=True prioritizes the complete last.pt; select a new RUN to change initial checkpoints.

Notebook 05 automatically rebuilds/cache text embeddings when a different frozen encoder is selected. Its bubble features, labels and original split remain unchanged. Notebook 07 recomputes text embeddings live.

Every best validation improvement saves RUN/best.pt and RUN/classifier.pt, plus individual classifier modules under RUN/classifier_layers/. Notebook 07 also saves RUN/deberta/ and all encoder layers under RUN/deberta_layers/. The root inference_config.py pairs the best classifier with its encoder. last.pt remains the complete resume snapshot. Printed inference paths point directly at the run root.

## File flow (shared paths)

Set `TRAINING_PATHS` in config.py once. `deberta_run` currently points to the existing runs_deberta_resume_v1; `classifier_run` is runs_classifier_resume_v1_tuned_Bert. These names are paths, not examples of runs that already exist.

1. Preparation wrapper and notebooks 01–04 write selected-OCR/bubble input caches under CONFIG['cache_dir']; they do not create a training run.
2. Notebook 07 writes best.pt, last.pt, classifier.pt, deberta/, layer files and inference_config.py under TRAINING_PATHS['deberta_run'].
3. Notebook 05 reads SOURCE_RUN = TRAINING_PATHS['deberta_run'], loads its deberta/ and classifier.pt, builds matching frozen text features, and writes TRAINING_PATHS['classifier_run'].
4. Notebook 06 evaluates classifier_run if its best.pt exists, otherwise deberta_run. Set RUN explicitly to evaluate another experiment. It uses that run’s paired best encoder rather than the original semantic cache.
5. Full-page test notebooks select the same classifier_run/deberta_run order and load its generated inference_config.py to pair classifier and encoder.

Initialization paths are checked before extraction. ResumeTraining=True restores existing last.pt; changing EPOCHS extends the same run. An old run without last.pt can initialize weights in a new run but cannot restore its optimizer. No training or feature extraction was executed during the file-flow repair.

Notebook 05 now snapshots its frozen encoder into its own RUN/deberta/ before exporting best.pt. Resuming 05 uses this local snapshot, keeping the pair independent of later 07 training. Selected-encoder text extraction caches each completed sequence so interruptions can resume.
