# ModelFormatter sequence

`three ordered images → VLM transcript → frozen text formatting model → JSON`

Run `01_Baseline.ipynb`, restart the kernel to free GPU memory, then run
`02_Train_Evaluate.ipynb`. The baseline uses the untouched local Qwen2.5-VL
model; the trainer learns transcript targets with a fresh LoRA adapter.

The VLM prompt is `prompts/transcript.txt`; the second model uses
`prompts/formatter.txt`. Targets contain `PAGE 1` through
`PAGE 3` headers and one `speaker<TAB>escaped text` line per bubble. Empty
pages retain their headers. A separate instruction-tuned text model converts
these to the original three JSON page arrays. Its prompt requires preservation
of transcription, identity, and order; formatting mistakes count in evaluation.
Only the VLM receives LoRA training. The formatter remains frozen.

Speaker attribution must happen in the VLM stage while the images are available.
The baseline preview validates page headers and speaker/tab/text lines, shows
the extracted character IDs and UNKNOWN count, and flags token-limit exhaustion.
A plain OCR transcript without speaker IDs is invalid, even if its text is
readable. The frozen text formatter cannot recover visual speaker information.
Prompt compliance and correct visual attribution are separate checks: a valid
transcript can still assign the wrong character. The untouched baseline may
fail these instructions; training teaches the required transcript targets.
After changing the shared transcript prompt, rerun the baseline before training
so the saved experiment configurations still match. Existing raw transcripts
must be regenerated to benefit from the new prompt.

The model formatter receives the original transcript without strict parser
prevalidation. After generation, validation requires the three-page JSON schema
and exact preservation of text, speakers, page assignment, and order. This check
ignores surrounding page-header whitespace, blank lines, and LF/CRLF differences;
dialogue whitespace and escapes remain significant. Missing or reordered page
headers, invalid dialogue, and content changes remain explicit failures. The raw
model response is retained even when transcript validation fails afterward.

Both notebooks keep the text model as the main pipeline and include a comparison
cell using `format_transcript(text, tolerant=True)` on the same saved transcripts.
Separate predictions, metrics, and per-sequence reports are saved under
`tolerant_parser/`. The strict parser remains the default for canonical target
round trips. Neither parser invents missing pages or repairs missing dialogue.

The objective is strong validation results from `TaskData/score.py`, especially
`text_order_score` and `balanced_joint_f1`, while retaining transcription
coverage and cross-page speaker consistency. Both notebooks export scorer-ready
`submission.jsonl`, held-out `references.jsonl`, and `score.json`. The trainer
compares official baseline and tuned scores. Malformed predictions are logged
in `invalid_predictions.json` and count as missing sequences in the official
score. Training selects checkpoints by validation loss; final quality is judged
by the task scorer, so lower loss alone does not establish an improvement.

`datasets/train.jsonl` and `datasets/validation.jsonl` are derived from canonical
labels using the existing seed-42 72/8 split. Each row retains canonical JSON,
the lossless transcript target, and repository-relative image paths. The
baseline re-exports these files from source labels before inference.
Test images are never included in training or validation.

Artifacts use `artifacts/formatter_lora`; predictions and metrics use
`outputs/formatter_base_baseline` and `outputs/formatter_lora_eval`.
Predictions save both `raw_transcript` and formatted `prediction`, with
`raw_formatter_prediction` and `formatter_error` for failures. Invalid formatter
JSON or schema is scored as an invalid output. Generation diagnostics refer to
the VLM transcript tokens. Existing task metrics score the formatted JSON.

Both notebooks use the same prompt, split, processor settings, and generation
settings. Direct JSON notebooks remain in `notebooks-modelJSON`.
GPU inference and training require the same CUDA/4-bit environment as those
notebooks and `MANGA_VLM_MODEL_PATH` for a non-default local model location.
Both notebooks default to `C:\MODELS\HF\Qwen2.5-VL-3B-Instruct` and
`C:\MODELS\HF\Qwen2.5-1.5B-Instruct`. Set `MANGA_VLM_MODEL_PATH` and
`MANGA_FORMATTER_MODEL_PATH` before starting the notebook kernel to override
these folders, or edit the setup cells in both notebooks. The formatter must
be a local instruction-tuned causal language model with a chat template.
No formatter weights are bundled. All model loads use local files only.
The models run sequentially: all validation transcripts are saved first,
then the VLM, processor, collators, and any trainer/optimizer state are released
before the formatter loads. The formatter defaults to CPU float32;
`MANGA_FORMATTER_DEVICE=cuda:0` can use GPU after the VLM is unloaded. CPU formatting
may be slow. Both notebooks require exactly matching model paths, prompts,
split, and generation settings, recorded in `experiment.json`.

## Run order

1. Start with a fresh kernel and run every cell in `01_Baseline.ipynb` to export
   the transcript datasets, preview a raw transcript, save all holdout transcripts,
   unload the VLM, format the saved transcripts, and report official scores.
2. Restart the kernel before training.
3. Run every cell in `02_Train_Evaluate.ipynb` to train a new transcript adapter,
   save the best validation-loss checkpoint, and compare official scores.

Raw VLM outputs are saved separately at `raw_transcripts/predictions.jsonl`
under each evaluation output folder. Final `predictions.jsonl`, `metrics.json`,
and per-sequence reports are produced only after formatting. The raw file is
retained if formatting fails. To retry Stage 2 from a fresh kernel, run only
the setup and dataset cells, then the Stage 2 formatting and results cells;
skip processor/model loading, raw preview, training, and Stage 1 generation.

Training reads canonical labels and derives transcript targets through
`FormatterSequenceDataset`; the exported dataset files provide an auditable
copy. `datasets/split_seed42.json` records the exact shared sequence split.
Do not load the direct-JSON adapter into this workflow: its target format differs.
