# QuoteCaster_Finetuning

Run train.ipynb in godfather.venv. You run model loading, smoke test and training. No classifier, projection head or clustering is added.

Requires local C:/MODELS/quote-caster-encoder and TaskData/development/labels.jsonl. No existing project caches or images are required. Creates own tokenized inputs and fixed 64/16 sequence split manifest. Character IDs are local to a sequence. UNKNOWN/NARRATION remain context, never supervised targets. Counts/eligibility are recomputed from actual data.

Local model README confirms last_hidden_state[:,0,:], raw CLS output, and single-string context [SEP] quote. Default NORMALIZE=False preserves this; if enabled, normalization applies consistently to mining/loss/validation. No pooler output or extra head. Freeze embeddings/layers 0-7/pooler; fine-tune layers 8-11. Context includes ordered page/dialogue markers but never GT speaker labels. Centered windows reserve the target quote first.

Online within-sequence nearest semi-hard mining, nearest hard fallback, otherwise skip. Triplet loss margin .3, p=2, eps=0 matches cdist. AdamW 2e-5 / .01 decay, max 10 epochs, patience 3, BF16 where supported, clipping 1. Sequence-average optimizer loss; logged loss weighted by triplet count. Optional accumulation corrects partial batches. Gradient checkpointing/non-reentrant and chunk size 4 reduce memory; gradients from all chunks remain connected. Smoke test performs backward but no optimizer update. Structural fallback is smoke-only. Scheduler uses maximum possible update count; skipped sequences do not advance it.

Baseline epoch 0 and all validation epochs evaluate all eligible within-sequence triplets/pairs, with sequence-macro metrics. Primary best-selection metric: triplet accuracy; pairwise AUC breaks exact ties. Track margin satisfaction, distance means/medians and valid comparison counts. Paired bootstrap resamples sequences. No clustering performance claim.

outputs/: split_manifest.json, dataset_summary.json, tokenized_sequences.pt, experiment_config.json, baseline_metrics.json, baseline_distances.npz, training_history.csv, validation_history.csv, validation_by_sequence.csv, best_validation_by_sequence.csv, best_distances.npz, final_comparison.csv, final_report.json, sequence_bootstrap.json, plots/, checkpoints/best and checkpoints/final. Best may remain epoch 0. Source pretrained weights never overwritten. Running all starts a fresh training run; no optimizer resume. Choose another OUTPUT to preserve previous runs.

Original inference reference: local README.md and https://huggingface.co/aNameNobodyChose/quote-caster-encoder. Full original training code is not distributed locally; margin, unfreezing and normalization toggle are documented experiment settings.
