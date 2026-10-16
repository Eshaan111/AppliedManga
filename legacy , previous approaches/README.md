# Pre-Magi projects

Moved here on 2026-10-16 at the user's request. These are the actual project folders with their existing code, data, local environments, model assets and saved outputs. No training, inference or model conversion was performed.

| Folder | Earlier approach |
| --- | --- |
| [1pageTest](1pageTest/) | Independent bubbles, OCR, faces/bodies, InternVL association, manual labels, LoRA and panel order |
| [manga_vlm_crashcourse](manga_vlm_crashcourse/) | Qwen direct transcripts, LoRA, page augmentation, formatter and OCR-hint variants |
| [MultiModelSystem](MultiModelSystem/) | DINO identity, independent face detection, Paddle joining and Qwen attribution |
| [MultiPageTillAssociationPipeline](MultiPageTillAssociationPipeline/) | Sequence preparation, reviewed page targets and page-wise association training implementation |
| [1pageTest-EVERYTHING-GOOD](1pageTest-EVERYTHING-GOOD/) | Preserved backup snapshot; not another independent experiment |
| [1pageTest-EVERYTHING-ULTRA-GOOD](1pageTest-EVERYTHING-ULTRA-GOOD/) | Preserved backup snapshot; not another independent experiment |

Read [15 pre-Magi approach families](1pageTest/PRE_MAGI_APPROACHES.md) and [the complete experiment inventory](1pageTest/EXPERIMENT_INVENTORY.md) for sources, results and lessons. The current [Magi pipeline](../Magi/paddleOCR-Magi/README_ARTIFACT_CONTOUR_FULL_INFERENCE.md) stays outside this archive.

## Reusing archived code

Relative sibling-project links in the experiment guides have been updated. Archived notebooks, configuration files, saved manifests and local virtual environments can still contain their original absolute paths. These historical files were preserved rather than broadly rewritten. Reconfigure paths and recreate/register moved environments before reuse. Saved notebook output is historical evidence, not an execution after relocation.

Magi, the shared Applied-Task/.venvAppliedTask environment, C:/MODELS and C:/Users/ESHAAN/BYTE/godfather.venv resources were not moved. No compatibility junctions were created at the old project paths.

Windows prevented renaming the active 1pageTest directory. Every item inside it was moved into legacy/1pageTest instead, leaving the original directory empty. The other five project directories were moved as whole folders. See [relocation_status.json](relocation_status.json). No running processes were stopped and no project files were deleted.
