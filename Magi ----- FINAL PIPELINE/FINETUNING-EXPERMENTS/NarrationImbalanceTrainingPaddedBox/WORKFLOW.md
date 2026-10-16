# Padded-box workflow

1. Reuse the original complete 3-page samples and saved train/validation partition.
2. Keep DeBERTa semantic (768), Magi DETR visual (256), original geometry (4) and tail (2) features.
3. Expand every original text bbox by crop_padding per side, clip to the page, and fit the RGB crop on a white square without center cropping.
4. Extract frozen local DINOv2-small CLS features (384); cache by sequence/box ID and preprocessing signature.
5. Add dino to each original sample; preserve labels, -100 context boxes, class counts, class-balanced weights and split.
6. Project branches to 256 + 256 + 128 + 32 + 16 = 688; fuse to 256; add learned page/position embeddings.
7. Train the 2-layer sequence Transformer and classifier with the same whole-sequence oversampling and masked weighted CE.
8. Evaluate, run five seeds, then run this experiment's optimization, architecture and joint Optuna studies.
9. Export BestParamTuning with the extra DINO preprocessing contract for future inference.

Nothing assumes that text outside a bubble must be narration. Padding adds appearance evidence to be learned from the binary training labels.
