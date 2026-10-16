# Validation of the padded implementation

Checks use synthetic sequences/images only; no project-dataset training or DINO extraction was launched during setup.

- All notebooks/Python sources compile; every code cell has a Markdown description and saved outputs are cleared.
- Padding fractions, clipping and aspect-preserving full-crop preparation were checked.
- The actual downloaded DINOv2-small model and processor loaded offline and returned 384-dimensional CLS features for a synthetic image.
- Both padded model variants passed forward/backward checks, including the new crop-projection gradients.
- Collation keeps real unlabeled boxes in attention and masks padded boxes; DINO padding uses zeros.
- Synthetic notebook 04 joins preserved the base split and weights; 05 trained/saved; 06 restored/evaluated.
- Synthetic 08, 09 and 10 studies exercised three-seed objectives, five-seed reports and total-budget resume behavior.
- BestParamTuning export included checkpoint crop_dim, feature_config, crop_features.py and preprocessing.json.

These checks validate implementation wiring, not narration quality. Run the notebooks on the real caches to measure improvements.
