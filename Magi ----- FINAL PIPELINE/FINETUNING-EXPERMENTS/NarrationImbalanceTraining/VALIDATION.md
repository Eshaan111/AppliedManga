# Checks performed for this remake

- All seven notebooks validated against nbformat and every code cell compiled.
- Inspected local source notebooks in `isNarratorTextMagi`, `TextRoleVL` and `Manga109Association`; checked extraction calls against `C:\MODELS\magiv2`.
- Inspected the real legacy cache: 1,490 labeled samples with 256-dimensional DETR embeddings.
- Confirmed the existing Godfather Python 3.11 kernel imports PyTorch 2.14.0+cu126, has CUDA available, and uses Transformers 4.52.4.
- Ran stages 00/01 with real local GT, images and raw OCR caches in an isolated temporary output directory: 80 full sequences, 2,183 boxes; 1,427 SPOKEN, 63 NARRATED and 693 ignored.
- Used synthetic embeddings to execute feature-cache reuse, feature joining, one complete classifier epoch on CUDA, checkpoint reload, evaluation and CSV/JSON export. No synthetic caches/checkpoints are included in the installed folder.
- Checked no image-hash overlap across splits, exact train-only class counts, padding invariance of real-box logits and zero supervised-loss gradients for ignored labels.
- The exercised split had 64 training sequences and 16 validation sequences. Original train counts were [1142, 50]; validation counts [285, 13]. These counts are not model-quality results.
- Confirmed evaluation exports all 419 real validation boxes, including ignored ones.

Actual DeBERTa extraction, full-dataset Magi extraction and real-feature training were not executed. Those are the notebook runs left to the user. Synthetic-feature training checks pipeline mechanics only.

## Magi alignment regression check

Reproduced the reported shortage: seq_ee880b9612205860 page 1 has 12 original boxes but the filtered page cache has 11 candidates. Ran actual local frozen Magi inference with the revised full-query extraction: all 12 boxes matched distinct queries out of 300, with minimum IoU 0.9647. Verified a filtered candidate shortage requests refresh. The revised loader disables redundant pretrained-backbone downloads and requires every model parameter to load from the local checkpoint. Existing successfully aligned page caches remain reusable. No target caches were deleted or dataset-wide feature extraction/training run.


## Notebooks 07 and 08

Validated both notebook schemas and compiled their cells/shared helper. Confirmed the model's optional dropout argument preserves default parameter initialization, state-dict keys/shapes and evaluation logits. Tested five one-epoch seed runs on synthetic cached sequences, aggregate mean/std/min/max/range, completed-run reuse, immutable saved samples, and split checks. Executed two actual Optuna trials against synthetic data with SQLite, checked the mean-over-seeds objective, total-trial budget, resume behavior and five-seed selected-configuration comparison. Tests ran in isolated temporary folders; no real five-seed experiment or hyperparameter search was launched. Optuna test dependencies were isolated in the workspace, not installed into the user's kernel.
