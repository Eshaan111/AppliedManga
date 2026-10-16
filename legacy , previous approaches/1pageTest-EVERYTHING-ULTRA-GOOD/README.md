# One-page manga test

Implemented from the supplied starter in `C:/Users/ESHAAN/BYTE/Applied-Task/1pageTest`.
The five notebooks are intentionally unexecuted. No models were downloaded and
no dependencies were installed during implementation.

## Environment selection

Stages 1, 3, 4 and 5 select `appliedtaskbyte`, backed by
`C:/Users/ESHAAN/BYTE/Applied-Task/.venvAppliedTask` (Python 3.14.2).
Stage 2 selects `onepage-ocr`, displayed as **.venvOCR (1pageTest OCR)**, backed by
`C:/Users/ESHAAN/BYTE/Applied-Task/MultiModelSystem/.venvOCR` (Python 3.11.9).
Both kernels are registered with Jupyter; the server can stay in `.venvAppliedTask`.

The OCR environment already has PaddleOCR 3.7.0 and PaddlePaddle GPU 3.2.0.
Both imports were checked successfully and Paddle reports one visible CUDA device.
No OCR model was initialized and no notebook was run. Do not install the CPU
`paddlepaddle` package alongside this environment's `paddlepaddle-gpu` package.
Paddle dependencies were removed from the main requirements because Python 3.14
is unsupported by PaddlePaddle's Windows wheels.

The shared environment also has Transformers 5.17.0, ipykernel 7.3.0 and
Pillow 12.3.0. The starter requirements constrain older versions and would
change those shared packages. Do not install that file wholesale; install only
missing packages. The association notebook uses the original InternVL custom
API intended for Transformers 4.x; its compatibility with the installed 5.x
version is unverified.

JupyterLab starts successfully. If it restores missing `module-1` files from
another project, use `jupyter lab --default-url=/lab/workspaces/1pageTest`.
The skipped language servers are optional editor integrations.

References: [PaddlePaddle Windows support](https://www.paddlepaddle.org.cn/documentation/docs/install/pip/windows-pip_en.html),
[JupyterLab workspaces](https://jupyterlab.readthedocs.io/en/latest/user/urls.html).

## Prepare once

Use the existing environment at
`C:/Users/ESHAAN/BYTE/Applied-Task/.venvAppliedTask` (Python 3.14.2).
Select **AppliedTaskByte** for stages 1, 3, 4 and 5, and
**.venvOCR (1pageTest OCR)** for stage 2. Restart JupyterLab if the new OCR
kernel does not appear.

```powershell
cd C:\Users\ESHAAN\BYTE\Applied-Task\1pageTest
..\.venvAppliedTask\Scripts\Activate.ps1
jupyter lab
```

For the missing character detector runtime and detector weights:

```powershell
python -m pip install onnxruntime
hf download PSImera/manga_bubbles_detect bubbles_detect.pt --local-dir C:/MODELS/manga_bubbles
hf download comicbox/comicapp-char-detector dass_char_xl_dcm.onnx --local-dir C:/MODELS/dass_character_detector
```

Use a PyTorch/torchvision build matching your hardware. The requirements use
CPU ONNX Runtime; for ONNX GPU, replace `onnxruntime` with a
compatible `onnxruntime-gpu` package rather than installing both.

The five stage notebooks select `TaskData/development/images/seq_71033fec4de494fc/01.png`
(page 1). Change SEQUENCE_ID and PAGE_NUMBER in each notebook to select another page.
The kernel must start in the project root or one of its five stage directories.
Adjust model paths in `models.json`. PaddleOCR downloads its language models
when initialized; InternVL3-2B downloads to the Hugging Face cache when loaded.
InternVL uses repository custom code via `trust_remote_code=True` and the pinned
Transformers 4.x version. Dependency and inference compatibility remain unverified.

## Run manually in order

1. `1_bubble_detection/bubble_detection.ipynb`: bubbles.json, bubble_preview.png.
2. `2_ocr_from_bubbles/ocr_from_bubbles.ipynb`: ocr.json, bubble_crops/*.png,
   bubbles_clean.json, ocr_clean.json, bubble_gt_labels.json.
3. `3_character_detection/character_detection.ipynb`: faces.json, bodies.json,
   characters.json, character_preview.png.
4. `4_annotate_page/annotate_page.ipynb`: annotated_page.png, annotation.json.
5. `5_bubble_character_association/association.ipynb`: association_raw.txt,
   associations.json.

All results live in `data/intermediate`. JSON stage files carry a page fingerprint
to reject results from another page. Stage 5 also verifies stage 4's input and image
fingerprints. Inspect previews after every stage. A changed page requires rerunning
the pipeline; changed detection/OCR outputs require regenerating the annotation.
Old preview/crop files are not deleted automatically.

Blue boxes label bubbles; red boxes label character instances. OCR appears in a
side legend. Face/body matching is a simple geometric heuristic, and IDs do not
establish identity across panels. The supplied DASS checkpoint was fine-tuned on
western comics, so inspect its manga detections carefully. UNKNOWN covers unseen
or uncertain speakers; malformed VLM output raises an error instead of fabricating
assignments. Check raw output before adjusting a prompt and rerunning association.

Sources: [DASS model card](https://huggingface.co/comicbox/comicapp-char-detector),
[InternVL3-2B model card](https://huggingface.co/OpenGVLab/InternVL3-2B),
[PaddleOCR 3 API](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html).

## Bubble cleanup and GT labels

Stage 2 removes overlapping duplicate bubbles only when their normalized OCR text
is identical. The larger box is kept; original IDs and raw files are preserved.
Stages 4 and 5 read bubbles_clean.json and ocr_clean.json. Character detection
remains stage 3.

The last OCR cell reads labels.jsonl for the selected sequence and page and picks
one best transcript match for each retained bubble. A score below 80 leaves
speaker=null and keeps the bubble. There is no multiple-match or split-transcript
handling. bubble_gt_labels.json records bubble -> OCR text -> GT speaker;
GT identities such as character1 still need to be connected to detected C IDs
before making a visual identity training target. GT labels are not supplied
to the inference prompt.

Rerun stages 1 through 5 manually to regenerate results under the selected
dataset filename and new cleanup flow. Existing results and previews have not
been regenerated by this edit.

## Independent panel reading-order experiment

7_panel_reading_order/panel_reading_order.ipynb adds local YOLO panel detection, Manga109 geometric panel ordering, center-based bubble assignment and approximate row-based bubble order. It reads cleaned stage-2 bubbles/OCR and writes separate results under data/panel_reading_order/<page_id>/. Unassigned bubbles and inseparable panel groups are flagged for review. It does not change speaker association or training.

Provide C:/MODELS/manga_panel_detector/manga_panel_detector_fp32.pt and the complete C:/MODELS/panel-order-estimator checkout yourself. The notebook was created without execution, model downloads, repository cloning or dependency installation. See the stage README for required local files and packages.
