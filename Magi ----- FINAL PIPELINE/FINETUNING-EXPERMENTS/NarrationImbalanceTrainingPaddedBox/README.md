# NarrationImbalanceTrainingPaddedBox

Same staged workflow as NarrationImbalanceTraining, with one added branch: padded text-box crops → local frozen DINOv2-small → 384-dimensional CLS features → trainable 128-dimensional projection. Existing DeBERTa, DETR, geometry and tail branches remain. Concatenation grows from 560 to 688 dimensions; fusion still produces 256 dimensions. All encoders stay frozen.

Run in a NEW/RESTARTED Godfather kernel in this folder so config/model imports do not come from the original notebook session. Edit config.py first. DINOv2 loads ONLY from C:\MODELS\dinov2-small with local_files_only=True; no downloads/installations or training are triggered by setup. Each code cell has a description and comments.

## Run order

00 → 01 → 02 → 03 → **03a_cache_dino_crops** → 04 → 05 → 06 → 07 → 08 → 09 → 10.

- 00 inspects paths/caches; 01–03 retain the original preparation and frozen feature workflow.
- 03a previews padded crops and extracts local frozen DINO features. Cache hits skip model inference; completed sequence features survive interruption.
- 04 joins DINO features to the original full sequence samples. It NEVER reruns splitting. The original training/validation IDs, labels, ignored boxes, class counts and effective-number weights are preserved.
- 05 trains a fresh padded classifier; 06 evaluates it. Existing classifier checkpoints have incompatible fusion/projection shapes and are not copied.
- 07 runs five initialization/dropout/sampling seeds on the same split.
- 08 tunes optimization/sampling settings; 09 tunes Transformer heads/feedforward width; 10 jointly refines around the winners saved INSIDE THIS folder. Prior experiment's study results are not imported.
- 08–10 retain their original configurable search spaces/budgets and final five-seed reports, graphs, and BestParamTuning export. In 09, TOTAL_TRIALS controls how much of its grid runs; increase it to cover every grid combination if desired.

## Padded crop contract

`crop_padding = 0.20` adds 20% of original width on the left/right and 20% of height above/below, clipped to image boundaries. A 100×50 box becomes up to 140×70 before resizing. Original boxes, OCR and geometry remain unchanged.

Use RGB. Resize the ENTIRE padded crop to fit a 224×224 white square, preserving aspect ratio. Disable the processor's additional resize/center crop so bubble borders are not discarded. Use its normal rescaling/normalization and DINO CLS output. No random augmentation or patch masking is used. This gives visual evidence, not a rule that non-bubble text is narration.

`crop_image_size`, `crop_batch_size`, padding and model path are configurable. Size must be divisible by local DINO patch size (14). Changing padding/model/preprocessing invalidates dino.pt and samples.pt. After training, use a new run/study folder when making those changes; guards reject stale checkpoints/results.

## Model dimensions

| Branch | Input → projection |
|---|---|
| DeBERTa | 768 → 256 |
| Original Magi DETR | 256 → 256 |
| New frozen padded DINO crop | 384 → 128 |
| Original geometry | 4 → 32 |
| Original tail features | 2 → 16 |
| Concatenate and fuse | 688 → 256 |
| Position/page embeddings and 2-layer Transformer | 256 → 256 |
| Classifier | 256 → 2 logits |

Loss, whole-sequence narration oversampling, padding attention masks and -100 ignored labels are unchanged. Gradient logs include the new crop projection. Run old and padded models on the same seeds/settings for a controlled comparison; comparing independently tuned winners also changes hyperparameters.

## Cache/data contracts

The small compatible caches are copied into this folder at setup: sequences.json, deberta.pt, magi.pt and **base_samples.pt**. The last is the original samples/split snapshot, not ready-to-train padded samples. No old runs, studies or BestParamTuning weights are copied. `base_cache_provenance.json` records their origins.

New `dino.pt` stores signature, preprocessing/model identity, vectors indexed by sequence/box ID and padded bounds. `dino_sequences/` stores resumable per-sequence caches. New `samples.pt` contains every original sample plus `dino: (N,384)`, feature_config and a new signature. Box IDs join modalities one-to-one. All caches/runs are independent of the old experiment.

The old split snapshot is required for this controlled experiment. If it is absent, point base_experiment to the original experiment; the notebooks copy missing base caches, but do not silently create a different split.

## Saved models and inference

Best checkpoints include crop_dim and feature_config. BestParamTuning adds crop_features.py and preprocessing.json to the checkpoint, model code and study metadata. Inference must provide the extra dino tensor and instantiate the padded model using crop_dim. The original full inference notebook is unchanged and cannot use this new model until that branch is added there.

See ORIGINAL_WORKFLOW.md for the original loss math, sequence/mask contracts and logging details. All original notebook names 00–10 are retained; 03a is the only additional stage.
