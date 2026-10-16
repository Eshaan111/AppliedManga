# OCR-supported manga workflow

`three ordered images + OCR hints ? VLM transcript ? frozen text formatter ? task JSON`

This experiment adds EasyOCR detections to the existing VLM prompt. The VLM
uses the images to group text into bubbles, filter non-story text, recover
reading order, and identify speakers across all three pages. The second model
only formats the transcript; it does not receive OCR hints or images.

## Run

1. In your existing manga environment, install the one additional dependency:
   `python -m pip install -r notebooks-OCRsupported/requirements.txt`.
2. Run `01_Baseline.ipynb` from a fresh kernel. Inspect the OCR preview before
   VLM generation, then run the remaining cells to score the eight held-out sequences.
3. Restart the kernel and run `02_Train_Evaluate.ipynb` to train a fresh LoRA
   adapter with the same OCR hints and compare against the OCR-supported baseline.

Both notebooks use `prompts/transcript.txt` and `prompts/formatter.txt` in this
folder. Model paths remain configurable through `MANGA_VLM_MODEL_PATH` and
`MANGA_FORMATTER_MODEL_PATH`. The VLM requires the existing CUDA/4-bit environment;
the text formatter defaults to CPU (`MANGA_FORMATTER_DEVICE` can override it).
The local defaults are `C:/MODELS/HF/Qwen2.5-VL-3B-Instruct` and
`C:/MODELS/HF/Qwen2.5-1.5B-Instruct`.

## OCR

`ocr_support.py` reads full-resolution images with EasyOCR on CPU before the
VLM loads. It defaults to English (`OCR_LANGUAGES = ["en"]`) for the translated
manga pages. EasyOCR downloads its recognition/detection weights on first use;
subsequent uncached runs can reuse the locally downloaded weights.

The hints contain page numbers, detected text, confidence, and pixel boxes
`[left, top, right, bottom]`, with the origin at the top-left. Detection order
is not treated as reading order. No boxes are automatically merged into bubbles
and no confidence cutoff drops detections. The prompt tells the VLM to verify
OCR against the images, recover omissions, and exclude sound effects and other
non-task text. OCR errors can still hurt results; inspect the preview and scores.

Detections are cached under `datasets/ocr_cache/`, keyed by image content,
language settings, and cache version. Training reuses validation detections
and computes training-page detections once. Delete the cache after changing
EasyOCR versions or detection settings. OCR consumes only image paths and bytes;
reference dialogue and speaker labels are never passed to it or to the VLM's
user message. Canonical labels supply unchanged assistant training targets and
scoring references.

## Outputs and evaluation

The seed-42 72/8 sequence split and canonical labels are shared with the existing
workflows. Dataset exports remain in this folder's `datasets/`. This experiment
uses separate outputs and adapters:

- `outputs/ocr_base_baseline/`
- `outputs/ocr_lora_eval/`
- `artifacts/ocr_lora/`

Raw VLM transcripts are saved under `raw_transcripts/predictions.jsonl` before
the VLM is unloaded and the formatter loads. Final predictions retain the raw
transcript, formatter response, and formatting errors. A tolerant parser
comparison is saved under `tolerant_parser/`; formatting failures still count.
The official unmodified `TaskData/score.py` reports `text_order_score` and
`balanced_joint_f1` under `task_score/`. Compare these scores with the original
`outputs/formatter_base_baseline/task_score/score.json` to assess OCR's effect.

Both notebooks record OCR settings, prompts, model paths, split, and generation
settings in `experiment.json`; training requires a matching baseline. After
changing the OCR settings or prompt, rerun the baseline before training.
All copied notebook outputs have been cleared so old results are not presented
as OCR results.

To retry only formatting in a fresh kernel, run the setup and dataset cells,
then Stage 2 and the scoring cells. Skip the OCR preparation, model loading,
preview, training, and Stage 1 generation cells. Transcript targets remain
`PAGE 1` through `PAGE 3` with one `speaker<TAB>escaped text` line per bubble.
Empty pages retain their headers; only the VLM is trained.

EasyOCR API reference: https://www.jaided.ai/easyocr/documentation/
