# Contour full inference: README and run guide

Start with [pipelineContourFullInference.ipynb](pipelineContourFullInference.ipynb). It produces speaker-linked dialogue using Magi, Paddle OCR, YOLO bubbles, contour narration rules and our trained character crop encoder.

- [Mermaid flowchart and implementation contracts](PIPELINE_CONTOUR_FULL_INFERENCE.md)
- [All fine-tuning experiments](../../FINETUNING-EXPERMENTS/README.md)
- [Standalone contour experiment](testContourNarrator2.ipynb)

## Full flow

1. Load sequence pages in filename order.
2. Reuse or run local Magi detections, OCR, character clusters and text-character links.
3. Reuse or run Paddle on original Magi text crops, then select the OCR string.
4. Delete artifacts before grouping: agreement strictly below 0.4, or SFX in Magi/Paddle/selected OCR. Missing agreement is retained unless SFX occurs.
5. Detect YOLO bubbles and match them only to surviving Magi texts.
6. Check individual boundaries; select overlapping groups containing at least one unresolved individual. Verified neighbors may participate; groups with every individual verified are skipped.
7. Concatenate original member texts in Magi reading order. Merged-boundary success is not required. Every surviving text appears exactly once; remaining texts stay standalone.
8. Inspect the combined bubble crop with 20% padding per side. Extract surrounding white/solid regions, retry uncertain extraction with faint barriers, and assign SPOKEN/NARRATED.
9. Resolve the group speaker from original member links, fine-tuned character embeddings and cross-page identities.
10. View annotated pages/groups and export transcript, predictions and audits.

Character/entity preparation runs earlier in the notebook for reuse; final speaker routing follows group narration labels. No GT labels are read for inference. OCR is concatenated as recognized: hyphens and reading order are not automatically rewritten.

## Pretrained, fine-tuned and heuristic parts

| Component | What runs here | Type |
|---|---|---|
| Text/character/panel/tail detection | Original local Magi | Pretrained; detector not fine-tuned here |
| Magi and Paddle OCR | Existing local models | Pretrained |
| Bubble detection | Existing local YOLO checkpoint | Loaded checkpoint; no training here |
| Character crop embeddings | Magi crop ViT with our final two blocks patched | **Our fine-tuned weights** |
| Text-character matching | Original Magi head by default | Pretrained; optional trained head disabled |
| OCR selection/deletion | Text validity, agreement and SFX rules | Heuristic |
| Character duplicate cleanup | Containment, IoU and cosine similarity | Heuristic using learned features |
| Cross-page identity matching | Best-member cosine, threshold 0.82 and per-page constraints | Heuristic using fine-tuned features |
| Grouping/concatenation | Text coverage, boundaries and overlaps | Heuristic |
| SPOKEN/NARRATED | Regions, smooth arcs, sharp edges and box sides | **Heuristic** |
| Final speaker | Narrator routing or unique member entity | Heuristic routing of model associations |

The learned narration classifier, DeBERTa, DINO and tail label override do not run here. A spoken label establishes a role, not the identity of the speaker. Alternative learned models are documented in the experiment index.

## Environment and required files

Use the existing **godfather.venv** Jupyter kernel:

`C:\Users\ESHAAN\BYTE\godfather.venv\Scripts\python.exe`

Keep the compatible Torch/Paddle CUDA installation; Torch imports before Paddle. No package installation/model download cells are provided. On cache misses, inherited Magi/Paddle inference requires the working CUDA environment.

| File/path | Purpose |
|---|---|
| `C:\MODELS\magiv2` | Original Magi weights and local code |
| `C:\MODELS\manga_bubbles\bubbles_detect.pt` | YOLO detector |
| `%USERPROFILE%\.paddlex\official_models\PP-OCRv5_server_det` | Paddle detection model |
| `%USERPROFILE%\.paddlex\official_models\en_PP-OCRv5_mobile_rec` | Paddle English recognition model |
| `FINETUNING-EXPERMENTS/fineTuningMagiCharSimHeads2/magi_crop_last_2_blocks.pt` | Trained character crop blocks |
| `TaskData/<split>/images/<sequence>/01.png`, `02.png`, `03.png` | Ordered input pages |
| [contour_narrator2.py](../contour_narrator2.py) | Plain-function snapshot of tested v2 rules |
| [contour_grouping.py](../contour_grouping.py) | Individual boundary checks and overlap proposals |

Keep the helper files beside the notebook. Edits to the standalone test notebook do not automatically update the helper snapshot; port a tested rule change to the helper too. Helper hashes participate in contour cache identity.

## Running

1. Open the notebook and select godfather.venv.
2. In Configuration, set MAGI_ROOT, SPLIT, SEQUENCE_ID and model/checkpoint paths.
3. Set PAGE_NUMBER=None for all three pages, or a number such as 2 for that page only. Single-page identities use only that page.
4. Leave ASSOCIATION_HEAD_PATH=None for original Magi associations. Enable a separately trained matching-head checkpoint only deliberately.
5. Keep ENTITY_THRESHOLD=0.82 and THRESHOLD_CHECKPOINT_EPOCH=20 paired with the current crop checkpoint. Its depth, epoch and base hashes are validated. A replacement checkpoint needs its own validated threshold/epoch.
6. Run cells top to bottom. CACHE/IMPORT/RUN messages show reuse or computation. After settings/model/helper changes, restart and rerun all cells.
7. Review annotated results, especially needs_review=True and unresolved speakers.

CONTOUR_SETTINGS and GROUP_FAINT_SETTINGS control individual checks/grouping. REGION_SETTINGS and FAINT_REGION_SETTINGS control narration regions/retries. DETECTION controls YOLO confidence/image size. FORCE flags can regenerate stages. Signatures/checksums/dependencies control cache invalidation.

## Narration rules

| Evidence | Role |
|---|---|
| Substantial smooth white arc | SPOKEN; overrides a box-like approximation |
| Actual straight sides in two perpendicular directions | NARRATED; rectangle/stepped caption evidence |
| Other meaningful curved/sharp white edges without caption evidence | SPOKEN |
| No qualifying white/solid surrounding region | NARRATED fallback |
| Unclear/leaking extraction or conflicting separate member regions | NARRATED + review |

Rectangularity is diagnostic, not a standalone caption decision. Crop borders never supply shape evidence; touching a crop edge alone is not rejection.

Smooth-arc evidence uses physical length relative to typical member text size, turning, bulge and directional consistency. It avoids a percentage of the entire merged perimeter, reducing dependence on horizontal/vertical layout and number of lobes. Exact thresholds remain editable heuristics.

Confident first passes are retained. Missing/uncertain/leaking regions and incomplete box-like evidence get one faint-barrier retry. Locally darker lines plus low-gradient edges block background connectivity; the retry does not close the white mask through faint outlines. Every group member must be adequately covered. Disconnected regions are checked separately; missing/conflicting member evidence is flagged.

## Speaker rules

| Group/member associations | speaker / status |
|---|---|
| NARRATED | NARRATION / narrator |
| SPOKEN, one entity and every member linked | Entity such as E1 / linked |
| SPOKEN, one entity with some unlinked members | That entity / partially_linked + review |
| SPOKEN, no entity | null / unassociated + review |
| SPOKEN, multiple distinct entities | null / conflicting_entities + review |

There is no majority vote, nearest-character guess or reassociation from the large merged text rectangle. Original member links are preserved. E1/E2 are sequence-local inferred identities, not names or global IDs.

A correct SPOKEN result can have an unresolved speaker. Cached PG2:M1 has speech-shape evidence but conflicting original member entity links; association uncertainty is separate from its narration label.

## Viewers

`show_page(page_index=0, associations=True)` displays groups and original character links. The index is zero-based within selected pages. Yellow means spoken, grey narrated, and optional red boxes are deleted OCR. Toggle member boxes, characters, panels, links, OCR and deleted boxes through widgets or function arguments.

`inspect_group('PG2:M1')` shows the crop, member boxes, region mask, faint barriers, visible edge sections, member candidates and final metrics. Use a group ID present in your run. Inspector edge colors represent curve/flat/sharp diagnostics, not final role colors. Inspection recomputes CPU contours only, not models.

## Caches and exports

New cache: `Magi/paddleOCR-Magi/cache_contour_full_inference/<split>/<sequence>/sequence/`.

For a selected page, sequence becomes page_02, etc. Results go in its outputs/ directory. Compatible original inference payloads are copied into the separate cache, without modifying source caches. YOLO reuse supports bubble JSON, page.pt feature caches and standalone test caches when identities match.

| Export | Contents |
|---|---|
| transcript.txt | One line per group/standalone text; UNKNOWN for unresolved speech; review flags stay in detailed audits |
| dialogue_pages.json | Ordered transcript with role, speaker, member IDs and review |
| group_predictions.json/.csv | Decisions, region/arc metrics and original member associations |
| box_predictions.json/.csv | Every original detection, deletion status and group/final mapping |
| deleted_ocr.json/.csv | Removed boxes and reasons |
| grouping_audit.json | Individual checks, merge proposals and ignored candidates |
| entities.json | Character identity matching audit |
| run_metadata.json | Inputs, checkpoint/settings identities and stage references |

The final cell includes an optional annotated PNG export loop. No classifier logits/calibrated narration probabilities are invented for these rules.

## Troubleshooting and interpretation

- **Visible bubble labeled box-like:** inspect mask spill into panels/background, whether retry ran, and smooth-arc metrics. A visible image edge can be absent from the extracted region.
- **Spoken but unresolved speaker:** inspect original member text-character links. Better contours do not fix an incorrect Magi association.
- **Missing dialogue:** inspect deleted_ocr and surviving member IDs. Deleted boxes intentionally cannot reappear through YOLO/grouping.
- **Wrong concatenation:** inspect overlaps, individual checks and member IDs. Connected overlap grouping is not a semantic guarantee.
- **Missing sequence:** check trimmed split/sequence and actual page filenames.
- **Checkpoint/hash assertion:** use the intended model and paired checkpoint/validated threshold rather than bypassing checks.

Cached-data checks cover exact-once membership, speaker routing, viewers, cache reuse/invalidation and serialization. They do not establish overall narration accuracy or replace a fresh GPU inference run and visual review.

## Schema-compatible transcript export

The final cells export transcript.json and transcript.jsonl in the exact development/labels.jsonl structure: one object with sequence_id and pages, with only speaker/text keys in each page row. Resolved predicted entities become stable sequence-local character1, character2, ... names; narrated groups use NARRATION and unresolved spoken groups use UNKNOWN. The numbering does not imply GT identity alignment. transcript_speaker_mapping.json retains the entity-to-character mapping; full review/association details remain in group/box prediction exports.

Original sequence page slots are preserved. In single-page mode, unprocessed pages are empty lists. The final notebook cell prints the entire sequence object. transcript.txt uses the same speaker names in readable form. Each run overwrites its own transcript.jsonl with one sequence line, avoiding duplicate entries on reruns.

## Final Gemma OCR-text cleanup

After grouping and speaker attribution, the notebook uses local C:/MODELS/gemma-3-4b-it in 4-bit with the approved few-shot OCR-restoration prompt. Each transcript text entry is corrected independently; speaker IDs, group membership, page order and entry counts are unchanged. Original OCR remains in transcript_raw.json/.jsonl/.txt and detailed group/box/dialogue exports. Final transcript.json/.jsonl/.txt and the final print cell contain cleaned text. transcript_cleanup_audit.json records raw/cleaned pairs, changes and empty-response fallback.

Compatible gemma_transcript stage caches skip model loading/generation. Input, prompt, model identity and algorithm changes invalidate cleanup separately from OCR/contour stages. The model is released after cleanup. Rerunning always starts from original group OCR, not previously cleaned text. Empty generated text falls back to original text; model edits still need review.
