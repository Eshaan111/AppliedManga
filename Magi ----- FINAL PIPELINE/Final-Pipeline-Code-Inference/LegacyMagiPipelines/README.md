# Legacy Magi pipelines and tests

Moved here on 2026-10-16. This folder contains the earlier notebooks, pipeline documentation and their caches formerly located directly in paddleOCR-Magi. They are preserved as experiment history, not the selected submission entry point.

The current pipeline remains at [pipelineArtifactContourFullInference.ipynb](../pipelineArtifactContourFullInference.ipynb). Its [guide](../README_ARTIFACT_CONTOUR_FULL_INFERENCE.md), [flowchart](../PIPELINE_ARTIFACT_CONTOUR_FULL_INFERENCE.md), [dependency manifest](../PIPELINE_ARTIFACT_CONTOUR_DEPENDENCIES.json), two contour helpers, audit tools and current cache remain in the parent folder.

Archived topics include learned narration/bubble/DINO pipelines, earlier contour pipelines, artifact tests, gibberish comparisons, OCR correction tests, character-cache and overlap experiments. See [the full experiment inventory](../../../legacy/1pageTest/EXPERIMENT_INVENTORY.md) and [fine-tuning index](../../FINETUNING-EXPERMENTS/README.md).

## Reuse

Archived notebook code and saved outputs were not rewritten. Absolute paths and cache settings can still refer to their original location; configure them before rerunning. The shared contour_narrator2.py and contour_grouping.py helpers are retained in the parent directory for the active pipeline. No models, data or saved outputs were deleted, and no inference/training was run.

[relocation_manifest.json](relocation_manifest.json) lists the 38 moved items and protected-code hashes. Earlier Markdown links and source-provenance paths were updated for this organization.
