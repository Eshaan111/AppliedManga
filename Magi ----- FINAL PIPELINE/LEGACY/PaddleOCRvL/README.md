# PaddleOCR manga page test

Open test_page_ocr.ipynb and run cells in order. Edit MODEL_DIR and PAGE_PATH in the first cell.
Default model location: C:\MODELS\PaddleOCR-VL-For-Manga. No download notebook is included.

Download separately in a terminal:

```powershell
python -m pip install -U huggingface_hub
hf download jzhang533/PaddleOCR-VL-For-Manga --local-dir "C:\MODELS\PaddleOCR-VL-For-Manga"
```

Use a working PyTorch installation with a suitable CUDA build, preferably a separate Python
3.10–3.12 environment, then install requirements.txt in that notebook environment. Restart the
kernel after installation. This experiment uses the original Transformers custom-code route
with trust_remote_code=True and local_files_only=True. It does not install packages automatically.

Test A feeds the full page directly to the recognizer. Test B optionally runs each existing Magi
text-box crop and compares it with cached OCR. The manga checkpoint was trained on crops, so
Test B is the more representative recognition test. Neither test establishes story eligibility.

The default image and raw-cache paths point to an existing local development page. When changing
PAGE_PATH, also update RAW_CACHE and PAGE_INDEX, or set RAW_CACHE=None for full-page-only testing.
MANUAL_BOXES supports another image without a Magi cache. Coordinates are xyxy in original pixels.
Images are displayed with EXIF orientation applied; manually configured boxes must match that orientation.

Outputs go to results/<page-stem> below the notebook's working directory. Each cache checks the
page hash, model config, OCR prompt, token budget, crop bounds, and numerical precision/device.
Set FORCE_OCR=True to regenerate, especially after replacing weights in-place. Output text is
preserved, including punctuation and line breaks. Full-page generation does not supply boxes or
guarantee reading order. Crop boxes and their order come from the configured cache.

The notebook has been schema/syntax checked and preprocessing/export logic tested using a stub
recognizer on the configured local page. Model weights were not downloaded and real OCR inference
was not run during creation. English-page accuracy is not established by Japanese manga training.

Sources:
- https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL


## Settings for the 8 GB RTX 4060

The original full-page run produced a CUDA out-of-memory error. Updated defaults skip full-page
generation and test only three cached text crops. Attention uses SDPA (supported by the installed
model code), max_pixels=401408, crop generation is capped at 128 tokens, and max_time=60 seconds
is a soft decoding budget. It cannot interrupt an expensive first forward pass. Text streams while
decoding and the notebook prints each crop and input token count before generation.

Interrupt an existing run, restart the kernel, and reload the updated notebook before running it.
Set MAX_CROPS=None for every box after the three-crop smoke test. Increase MAX_PIXELS or token
budgets if outputs lose small characters or are truncated. Existing other GPU workloads may still
cause memory pressure. The original notebook is saved as test_page_ocr.before_speed_fix.ipynb.
The updated settings were syntax/schema checked; no new model instance was loaded while your
GPU was busy, so an actual inference speed improvement has not yet been measured.
