# ContourNarrationMLP

Run `TrainContourNarrationMLP.ipynb` in godfather.venv. It is the only notebook and contains data preparation, cache generation, training and evaluation. Codex has not run extraction or training.

Set paths/settings at the top. Reuse `NarrationModelwBubbles/cache_selected_ocr/sequences.json` (matched GT/OCR) and `samples.pt` (fixed 64/16 split). Source caches are read-only. Magi tails and YOLO bubbles are reused or generated locally. Local frozen DeBERTa embeds previous two same-page texts + current text. No downloads.

MLP inputs: current + previous-two contour numeric measurements/availability flags, context-presence flags, and DeBERTa 768 -> 128 text projection. Combined -> 128 -> 64 -> 2 logits, SPOKEN/NARRATED. No tail flags or heuristic roles enter the MLP. Tail-linked units bypass it and force SPOKEN. They can supply previous-text context. A merged unit with any linked member also bypasses it.

GT labels: existing confidently matched character boxes -> SPOKEN, NARRATION -> NARRATED. Unknown/unmatched or mixed/partially unknown merged groups -> ignored. Every supervised target must have no tail. This notebook uses the existing labelled OCR cohort; it does not run Gemma artifact filtering. Do not interpret metrics as full OCR/artifact-pipeline accuracy.

Oversampling: whole narration-containing training sequences sampled with replacement, default weight 2.5 versus 1. No internal box duplication. Effective-number class weights are computed from original no-tail training counts only. Validation is never oversampled. Feature normalization is fitted on labelled no-tail training targets only. Shared page hashes across split boundaries are rejected.

Run top to bottom. Preparation can take time: caches resume completed sequences. Training cell starts a new run; choose a new RUN directory to preserve an older run. Best checkpoint selected by no-tail validation macro F1. Outputs: cache/, runs/baseline_seed42/{best.pt,last.pt,history.csv,config.json,split.json,evaluation.json,val_predictions.csv,val_predictions.json}. Checkpoint includes feature contract, scaler, class order and model arguments. It must be used with the same contour helper, semantic encoder and feature ordering.

Reports: per-class precision/recall/F1/support, macro F1, balanced accuracy, train/validation loss, true-class probability summaries, gradient norms, heuristic comparison and combined tail-routing metrics. All ignored/unlabeled predictions remain in the audit. No Optuna, Transformer or encoder fine-tuning is used.
