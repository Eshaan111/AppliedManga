# Manga face pair experiment

Exactly two models, one job each:

`manga page -> hysts face detector -> face crops -> DINOv2-small -> cosine similarity -> manual same/different check`

- Face detection: [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector), YOLOv3 only. Weights are from the author's `hysts/anime-face-detector-yolov3` repository. Loads the face detector directly so HRNet landmarks are never downloaded or loaded.
- Embeddings: [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), normalized 384-dimensional CLS vectors. Already local at `C:/MODELS/dinov2-small`.
- No clustering in this workflow. The immediate question is whether cosine similarity separates same and different manga characters.

## Run from the workspace root (PowerShell)

The existing CUDA Python environment is used. The hysts package is installed in this experiment's `vendor/` folder so the shared environment remains intact. For another environment use Python 3.12+ and `pip install -r requirements-detection.txt`.

```powershell
$DinoPython = '..\.venvAppliedTask\Scripts\python.exe'
& $DinoPython DINO-EXPERMINTATOIN-IDENTITY\download_detector.py
& $DinoPython DINO-EXPERMINTATOIN-IDENTITY\detection.py --pages 3 --output DINO-EXPERMINTATOIN-IDENTITY\outputs\faces_run1
& $DinoPython DINO-EXPERMINTATOIN-IDENTITY\automatic_identity.py --detections DINO-EXPERMINTATOIN-IDENTITY\outputs\faces_run1\detections.json --output DINO-EXPERMINTATOIN-IDENTITY\outputs\pairs_run1
```

Use new output directories for each run. Detector weights download locally to `models/anime-face-detector-yolov3/model.safetensors`. Detection and embedding inference use local files. `download_model.py` downloads DINO if needed.

Alternatively run `notebooks/02_character_detection.ipynb`, then `notebooks/03_dino_identity.ipynb` with the AppliedTask CUDA kernel.

Continue with `notebooks/04_pair_review_and_score_analysis.ipynb` for the explained human-review workflow. It starts with the seven saved `two_model_pairs` pairs, generates a score-hidden review with full-page context, optionally expands detection/embedding inference across calibration sequences, samples 100 candidates toward 50–100 confident human labels, imports labeled CSV downloads, and compares same/different score distributions. The threshold stays at 0.8 and original artifacts are preserved. No model runs or human labels are supplied in the delivered notebook.

Its reusable code and notebook generator are also in `notebooks/`: `pair_review_workflow.py` and `build_pair_review_notebook.py`. Run the latter to regenerate notebook 04 without executing it. Review HTML must be opened directly from disk to display local page/crop images; download edits before closing, then import them in the notebook.

`annotations/characters.json` supplies the selected pages. Detection ignores its boxes and identity labels, so you can continue annotating independently. Your annotations and annotator are preserved. Detector crop IDs belong to a particular run; do not copy manual crop labels onto detector crop IDs without matching them visually.

## Inspect pairs

Detection writes overlays, exact face crops, `detections.json`, a blank `identity_labels.json`, and a detection review file. Inspect overlays for misses and false positives.

DINO writes:

- `embeddings.npy` and `crop_index.json`.
- `cross_page_pairs.json`: raw cosine scores and page/crop paths, within each sequence only.
- `pair_review.html`: side-by-side crops, up to 200 pairs, sorted by score within each sequence.
- `pair_review.csv`: all pairs with blank `same_character` and `notes` columns for human review. Enter `same` or `different`; leave uncertain cases blank.
- Per-sequence cosine matrices/heatmaps, configuration and summary.

The default 0.8 threshold produces only an exploratory `probably same`/`probably different` suggestion. Similarity is not a probability and the threshold is not validated. Check both high-scoring different-character pairs and low-scoring same-character pairs before deciding whether DINO works.

For exactly one saved pair:

```powershell
& $DinoPython DINO-EXPERMINTATOIN-IDENTITY\compare_pair.py --run DINO-EXPERMINTATOIN-IDENTITY\outputs\pairs_run1 --sequence SEQUENCE_ID --face-a p0_c000 --face-b p1_c000
```

For optional fixed-threshold metrics, fill the detector run's `identity_labels.json` with `{"sequence_id": {"p0_c000": "A", "p1_c000": "A", "p1_c001": "B"}}`, then pass `--labels <path>` to a fresh DINO run. Labels restart per sequence; blank labels are excluded. Pair-review CSV labels are for manual inspection and are not automatically consumed by this flag. Reserve separate heldout sequences before tuning thresholds. Without human labels there is no measured answer yet to whether DINO works well enough.

Older manual crop benchmark files remain available to preserve existing work, but are not part of these two notebooks or the pair-only commands above.
