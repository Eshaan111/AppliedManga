# Manga109 speaker association — Experiment 1

Run the five notebooks in numeric order. Code uses plain functions and single-line comments. `common.py` holds shared parsing, plotting, local loading, and evaluation functions; extraction and training stay visible in the notebooks.

1. In **01**, edit `BASE_XML_DIR`, `IMAGE_DIR`, and `DIALOG_XML_DIR`. Point at the directories containing book XML files and images organized as `images/BookName/003.jpg`. The default dataset paths are placeholders. Inspect the observed speaker reference types and 30 plotted pages, then set `VISUAL_JOIN_LOOKS_CORRECT = True`.
2. **02** saves the seeded 80/20 book split and preserves unlabeled text separately.
3. **03** loads `C:\MODELS\magiv2`, extracts frozen features, reports matching coverage for texts, bodies, unique speaker appearances, and complete text-speaker links, then plots IoUs and detection overlays. Review coverage before training. Defaults match local Magi's text/character detection thresholds, with IoU ≥ 0.5. Duplicate GT-to-query matches are excluded rather than assigned contradictory supervision.
4. **04** saves training pairs and fixed validation candidates, benchmarks the untouched head first, then trains only the existing matching head. Training keeps the nearest wrong character. Validation uses every matched body appearance on the page. The best checkpoint is selected by validation Top-1 accuracy.
5. **05** compares both heads against the same saved candidates and reports non-nearest and close-distractor accuracy, Top-2, ROC-AUC, coverage, score distributions, and about 20 visual examples.

Use your existing CUDA-enabled Magi notebook environment. Required imports: `torch`, `transformers`, `timm`, `einops`, `pulp`, `scipy`, `shapely`, `numpy`, `Pillow`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, and Jupyter/IPython. The local Magi code uses older Transformers internals, so use a kernel where that model already works. These notebooks do not install packages or download models/datasets.

OCR initialization and pretrained backbone initialization are disabled in the in-memory model configuration to prevent nested downloads. All detection and text-character association weights load from the local model. Crop architecture stays intact to avoid unrelated checkpoint shape changes; neither OCR nor crop inference runs. Missing OCR-weight warnings can occur because OCR is disabled.

Outputs are under `data/` and `checkpoints/`. Only head weights are saved; the local Magi model files are never edited. Cache checksums and extraction/candidate hashes prevent mixing experiments. When changing data, model files, or thresholds, rerun notebooks 03–05. Model file size/modification times and source/config hashes are recorded; keep the local checkpoint fixed during the experiment.

To apply the saved head to a loaded copy of the same local Magi model:

```python
checkpoint = torch.load(ROOT / "checkpoints" / "best_association_head.pt", map_location="cpu", weights_only=True)
model.text_character_matching_head.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

Accuracy is measured only on successfully matched, explicit visible-speaker links. Unresolved/off-page speaker IDs and ambiguous multi-speaker links are listed in `data/join_issues.json`. Body IDs refer to appearances, not persistent identities. Unlinked text is never labeled narration or used as an all-negative training example. Independent sigmoid scores allow low scores for all candidates, but the ranking benchmark assumes a visible labeled speaker and does not validate abstention.

The held-out books are validation books used for checkpoint selection, not an independent test set. No decoder training or Experiment 2 is implemented.

Dataset references: [Manga109 annotations](https://manga109.github.io/manga109-project-website/en/annotations.html), [Manga109Dialog public annotations](https://github.com/manga109/public-annotations), and [official Manga109Dialog project](https://github.com/liyingxuan1012/Manga109Dialog). Respect the datasets' terms and citation requirements.
