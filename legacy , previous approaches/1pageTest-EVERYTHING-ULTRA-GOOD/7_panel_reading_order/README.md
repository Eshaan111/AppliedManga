# Panel and bubble reading order

Open panel_reading_order.ipynb with AppliedTaskByte (.venvAppliedTask). Run manually after stage 2 has saved bubbles_clean.json and ocr_clean.json for the same page. The default page is TaskData/development/images/seq_71033fec4de494fc/01.png.

Supply the local files yourself:

```text
C:/MODELS/manga_panel_detector/manga_panel_detector_fp32.pt
C:/MODELS/panel-order-estimator/order_estimator.py
C:/MODELS/panel-order-estimator/manga109utils.py
```

The local repository import requires manga109api; detector inference requires an Ultralytics release supporting YOLO26. Install missing dependencies yourself in .venvAppliedTask. The notebook contains no install/download commands and never runs the estimator's dataset scripts. No Manga109 dataset is required for these calls.

The [detector](https://huggingface.co/leoxs22/manga-panel-detector-yolo26n) supplies class 0 panels. The [estimator](https://github.com/manga109/panel-order-estimator) orders their boxes. Inseparable groups are flagged, with a deterministic provisional order inside them. Both local Python files are required by the upstream API.

Bubble centers are assigned to containing panels; the smallest containing panel wins on overlap. Unassigned bubbles are preserved for review. PANEL_OVERRIDES lets you manually assign a B ID to a PN ID. Bubble order uses approximate top-to-bottom rows and right-to-left order within each row; adjust ROW_TOLERANCE for the page scale and inspect the preview.

Results are written only when you run the notebook, under data/panel_reading_order/<sequence>_page_<number>/:

- panels.json
- panel_order.json
- bubble_panels.json
- reading_order.json
- reading_order_preview.png

JSON files retain the page fingerprint. reading_order.json also records settings, source input/model/code hashes, unassigned bubbles and ambiguous panel groups. This stage does not change existing detections, labels, association results, training samples or VL images. Original B IDs are retained for a later transcript join.

The notebook was created without executing cells or loading/downloading models. Ordering and detector inference remain unverified until you run it with your local files.
