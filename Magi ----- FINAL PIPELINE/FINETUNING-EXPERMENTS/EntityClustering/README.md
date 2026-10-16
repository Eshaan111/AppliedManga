# EntityClustering

Run EntityClustering.ipynb in godfather.venv. Reuses the current full pipeline's Magi extraction, two-block fine-tuned character encoder, duplicate cleanup and greedy cross-page entity clustering. Own caches; no Paddle/Gemma/contour/narration models. Configure sequence or INPUT_IMAGES at the top. You run extraction; Codex only checked code/schema.

src/entity_clustering_core.py contains unchanged cleanup/clustering functions extracted from paddleOCR-Magi/pipelineArtifactContourFullInference.ipynb. Model loading/crop functions are reused inline. models.json records their source and current/previous checkpoint paths; weights are referenced, not copied. Current model is epoch-20 final-two-block crop encoder, 768d normalized embeddings, entity threshold 0.82. Original Magi supplies within-page clusters and text-character links before fine-tuned crop encoding.

Previous identity experiments registered: character identity MLP, residual projection, final-one-block crop tuning, final-two-block crop tuning. Registry also records frozen DINOv2-small and DASS detector resources used elsewhere; those are not active here. Historical model adapters/training are not rerun or automatically applied. Their original folders retain training/test notebooks and weights.

Outputs: cache/<sequence>/<signature>/outputs/entities.json, character_cleanup.json, models.json, run_metadata.json. Viewer shows stable entity colors and same-entity lines. This notebook does not train a new model or assign real character names.
