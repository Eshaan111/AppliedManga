# Legacy page-order augmentation

These files are retained only to preserve the history of the earlier page-
permutation experiment. They are not part of the active training or evaluation
pipeline.

The active source of truth is:

```text
TaskData/development/labels.jsonl
TaskData/development/images/<sequence_id>/01.png..03.png
```

Do not import or regenerate `augmented_labels.jsonl` for current runs.

`LORA_augmented.ipynb` is preserved to document the retired experiment. Its
old model checkpoints have been removed and must not be used as the clean
baseline for the canonical-only pipeline.
