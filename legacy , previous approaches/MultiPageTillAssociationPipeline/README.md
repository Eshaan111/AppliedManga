# Multipage preparation and one-page VL training

One sample is one annotated page plus OCR and the joint assistant JSON. No model or notebook has been run during project creation.

## Run order

1. `00_split_data.ipynb`: save the fixed sequence split once.
2. `01_bubble_detection.ipynb`: detect bubbles, using `.venvAppliedTask`.
3. `02_ocr_from_bubbles.ipynb`: run OCR, using your `.venvOCR` environment.
4. `03_cleanup_and_gt.ipynb`: remove duplicates/optional empty OCR and match GT.
5. `04_character_detection.ipynb`: run the existing DASS detector.
6. `05_annotate_pages.ipynb`: save page previews.
7. Run the labeling server and review pages. Save groups, rejected appearances, and unmatched/incorrect bubble speakers. Add missing appearances by box coordinates when needed.
8. `6_lora_training/00_make_assistant_targets.ipynb`: export all complete reviewed pages; inspect `export_report.json` for skipped pages.
9. `6_lora_training/TrainingWeightedTokens.ipynb`: train on all exported training samples, one page at a time. Unexported pages are counted; finish labeling/exporting first if you want the entire split.
10. `6_lora_training/ValidateWeightedTokens.ipynb`: generate on reviewed held-out validation pages.

## Data and models

Put the dataset at `TaskData/development/labels.jsonl` and `TaskData/development/images/<sequence>/01.png`, etc. `project.json` first checks this local folder and otherwise references the existing `1pageTest/TaskData/development` without copying data or results. Change that configuration before creating the split if needed.

Model paths in `models.json` are reused. InternVL stays at `C:/MODELS/InternVL3-2B`. No models are downloaded by project creation. PaddleOCR uses the same installed OCR environment and model-cache behavior as the original notebook; it may obtain missing OCR assets when you run it yourself.

## Fixed splits and caching

`data/splits/manifest.json` is created once, including sequence membership and input hashes. Later notebooks reuse it. Changing the seed/fraction after creation does not reshuffle it. Use a new project/data directory to intentionally define another split.

All preparation outputs live at `data/intermediate/<train|val>/<sequence>/page_NN/`. JSON wrappers include sequence, page, split, image path/hash, settings and dependency hashes. `FORCE_RERUN=False` reuses matching stage results. Missing or stale results are recomputed only when you run the notebook. Changes to cleanup settings reuse cached raw bubbles/OCR. Model hashes invalidate their stage cache if weights change; increment the visible stage `version` when editing processing code. Saved annotations are protected from changed preparation; move the affected response aside explicitly before relabeling.

Manual responses and exports have the same split/sequence/page organization. Target export is repeatable; changed exports require `FORCE_REEXPORT=True`. Training refuses stale exports and loads only pages approved by the fixed manifest. Each run saves a copy of the manifest and the exact training sample paths. Use a new run name for another experiment.

## Labeling server (PowerShell)

```powershell
& "C:/Users/ESHAAN/BYTE/Applied-Task/.venvAppliedTask/Scripts/python.exe" "C:/Users/ESHAAN/BYTE/Applied-Task/MultiPageTillAssociationPipeline/manual_labels/server.py"
```

Open http://127.0.0.1:8766. Choose train/val and a page, or use Previous/Next. Unprepared pages are disabled. Unsaved navigation prompts before discarding edits. Save is per page. Blue GT names are only for labeling; training images contain B/A IDs only. Appearance and bubble IDs are shuffled consistently in exported images/prompts/targets; `id_maps` records the translation. `P` identities are local to each page.

The second weight configuration gives all assistant JSON and end tokens weight 1; the original configuration remains selectable. The existing attention LoRA (last 8 layers, rank 4), frozen vision/projector/base weights, CUDA loading and gradient accumulation are reused. Epochs and learning rate are editable. Validation compares identities by appearance sets so arbitrary P numbering does not count as an error.
