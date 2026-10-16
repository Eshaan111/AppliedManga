# Generated datasets

Source of truth: `../TaskData/sequences.json` and `development/labels.jsonl`.
Run `python scripts/prepare_datasets.py` from the codebase folder to validate/rebuild
manifests against the persisted split. No image copies or fabricated labels are created.

| File | Purpose |
| --- | --- |
| train.jsonl | 64 sequence SFT examples with three ordered images, prompt, target |
| val.jsonl | 16 held-out sequence examples; target is never used in inference |
| test.jsonl | 15 unlabelled triplets; no targets |
| train_references.jsonl, val_references.jsonl | Official scorer format |
| components/transcription.jsonl | 48 validation page transcription tasks |
| components/reading_order.jsonl | 48 pages with shuffled candidates and arbitrary IDs |
| components/speaker.jsonl | 16 triplets with known text/order for speaker-only tests |
| annotations/character_crops.jsonl | Visual crop annotation template, boxes initially null |
| annotations/identity_pairs.jsonl | Initially empty; human-reviewed same/different crop pairs |
| annotations/ocr_filtering.jsonl | Preserved candidate-level review template; unset judgments stay null |

`images` paths are relative to `TaskData/`. `page_index` and `line_index` are zero-based.
The prompt and target are stored separately so reference answers cannot enter inference.
The label hash, seed, and validation ratio live in `../outputs/split.json`.
Rebuilding preserves this split and existing crop annotations. Changing the source
labels or split configuration requires deliberately archiving the existing outputs.

Reference crop annotations are preserved for future component training and calibration.
