# Approaches before the Magi-based pipeline

Reviewed 2026-10-16. These archived projects are under `C:/Users/ESHAAN/BYTE/Applied-Task/legacy/`, **outside Magi**. This records the earlier approaches the user identified, their concrete implementations, and what the available artifacts establish. The ordering below groups approaches by design; it does not invent an exact date-by-date chronology. Backup folders are preserved snapshots, not extra independent experiments.

We did train model adapters before using the Magi-based system. The verified examples are **fine-tuning pretrained models with LoRA**, not training a complete foundation model from scratch. Some additional encoder/projector ablations were implemented but have no confirmed completed run.

## 1. Direct images to transcript

**Approach:** Give Qwen2.5-VL-3B three ordered page images and ask it to return the entire transcript with speaker IDs and page structure. Compare the pretrained baseline with a trained LoRA branch.

**Sources:** [baseline notebook](../manga_vlm_crashcourse/notebooks-modelJSON/01_Baseline.ipynb), [training/evaluation notebook](../manga_vlm_crashcourse/notebooks-modelJSON/02_Train_Evaluate.ipynb), [failure log](../manga_vlm_crashcourse/notebooks-modelJSON/FAILURES.md). Local `manga_vlm_crashcourse/artifacts/canonical_lora/` and baseline/adapter evaluation output folders exist; this does not by itself establish every planned comparison completed.

**Observed limitation:** The saved baseline flattened the required three-page structure into a fenced list. Four of eight sequences hit the 1,024-token generation limit, with repeated dialogue/increasing character labels in examples. These are baseline observations, not proof that every LoRA output failed identically.

**Lesson:** Reading text, grounding speakers, maintaining cross-page identity, preserving reading order and producing exact JSON are separate problems. One generative answer can fail at any of them. Score coverage and format failures rather than reporting accuracy only on surviving outputs.

## 2. Page-order augmentation for transcript training

**Approach:** Permute the three pages and consistently reorder/relabel targets to create additional LoRA examples.

**Sources:** [legacy augmentation notebook](../manga_vlm_crashcourse/legacy/augmentation/LORA_augmented.ipynb), [legacy status](../manga_vlm_crashcourse/legacy/augmentation/README.md).

**Outcome/lesson:** Retired from the canonical pipeline; old checkpoints were removed according to the project guide. Reordered examples are not new independent manga data, and sequence identity/target alignment must remain consistent. No controlled gain or specific causal failure is established by the available retirement note.

## 3. Separate transcription from JSON formatting

**Approach:** Qwen2.5-VL produces a text transcript with page headers and speaker IDs; frozen Qwen2.5-1.5B turns it into the required JSON. A deterministic tolerant-parser comparison is also implemented.

**Sources:** [ModelFormatter guide](../manga_vlm_crashcourse/notebooks-ModelFormatter/README.md), [baseline](../manga_vlm_crashcourse/notebooks-ModelFormatter/01_Baseline.ipynb), [training branch](../manga_vlm_crashcourse/notebooks-ModelFormatter/02_Train_Evaluate.ipynb).

**Outcome/lesson:** Baseline outputs exist. Separating formatting makes errors easier to inspect, but a formatter cannot recover text or visual speaker evidence lost in the first stage. Exact preservation of content, page assignment and ordering must be checked; parseable JSON is insufficient. No aggregate improvement is claimed here.

## 4. OCR hints alongside page images

**Approach:** Add full-resolution EasyOCR text/boxes to the three-image VLM prompt, then use the same frozen formatter. Separately compare Qwen on an annotated single page with and without OCR hints.

**Sources:** [OCR-supported workflow](../manga_vlm_crashcourse/notebooks-OCRsupported/README.md), [OCR baseline](../manga_vlm_crashcourse/notebooks-OCRsupported/01_Baseline.ipynb), [single-page comparison](../MultiModelSystem/manga_notebook_codebase/notebooks/bubble_speaker_attribution/single_page_ocr_comparison.ipynb).

**Outcome/lesson:** The OCR-supported training branch is implemented with execution unconfirmed; the single-page comparison has saved outputs. Hints separate recognition from reasoning, but noisy hints can still be copied. They do not automatically establish the speaker. A general benefit was not verified in this review.

## 5. Independent bubble detection and crop OCR

**Approach:** Run the PSImera YOLO bubble detector, crop each candidate, then run PaddleOCR. Remove overlapping duplicates only when normalized OCR matches; retain original IDs and compare accepted text with reference labels for supervision.

**Sources:** [one-page project guide](README.md), [bubble detector](1_bubble_detection/bubble_detection.ipynb), [bubble OCR](2_ocr_from_bubbles/ocr_from_bubbles.ipynb).

**Observed limitation:** Empty OCR and duplicate/overlapping candidates required cleanup and visual checks. The early GT matcher selects one best transcript match per bubble and has no split/joined-transcript handling. Matching a GT speaker name to OCR does not identify a visible character instance.

**Lesson:** A YOLO bubble rectangle is a candidate, not a final text row or verified speech/narration label. Keep detection, recognition, deduplication and speaker attribution separate. These ideas later remained useful around Magi.

## 6. Independent face/body detection

**Approach:** Run the DASS ONNX detector, separate face/body outputs with class-wise NMS, and pair detections geometrically to create character appearances.

**Sources:** [character detection](3_character_detection/character_detection.ipynb), [model and pairing notes](README.md).

**Outcome/lesson:** Saved outputs exist. The project guide notes that this checkpoint was adapted to western comics, so manga-domain behavior needs inspection. Geometric face/body pairing is a heuristic; appearance IDs do not establish cross-panel identity. No detector benchmark failure is asserted from domain difference alone.

## 7. Annotated page plus InternVL speaker association

**Approach:** Paint bubble IDs and character boxes on the page, provide an OCR legend, and ask InternVL3-2B to assign each bubble to a character.

**Sources:** [page annotation](4_annotate_page/annotate_page.ipynb), [association](5_bubble_character_association/association.ipynb), [recorded failure notes](6_lora_training/README.md).

**Observed failure:** The recorded answer mapped B1 to C1 through B7 to C7, then repeated C7 for B8-B15. This is an ID-copying/repetition pattern to investigate, not usable supervision or proof of image grounding.

**Lesson:** Arbitrary ID compliance and actual visual association are different. A plausible complete mapping can still be wrong. Missing/off-panel speakers and overlapping bubbles need explicit handling rather than forced nearest-character assignments.

## 8. Manual review and randomized joint supervision

**Approach:** Build a local review website for appearance groups, rejected detections and corrected bubble speakers. Export shuffled appearance/bubble IDs and a joint characters/rejected_appearances/bubble_speakers target. Use grouped splits and assistant-only token supervision.

**Sources:** [manual review guide](manual_labels/README.md), [target maker](6_lora_training/00_make_assistant_targets.ipynb), [training guide](6_lora_training/README.md).

**Outcome/lesson:** Annotation/target implementations and saved preparation outputs exist. GT transcript identities cannot be silently equated to detector C/A IDs; human-reviewed mappings are necessary. Randomizing IDs targets copying shortcuts. Manual preparation is not evidence of a generalizing trained model.

## 9. Actual InternVL LoRA training: decision-heavy token loss

**Approach:** Fine-tune the language model attention adapters while freezing vision/projector/base weights. Put weight 1 on decision IDs, 0.1 on JSON structure/end tokens, and zero on fixed field names in the first configuration. This was a reviewed-page smoke/overfit experiment.

**Sources:** [preserved executed first configuration](6_lora_training/TrainingWeightedTokens_1stWeightConfig.ipynb), [loss/training implementation](6_lora_training/weighted_tokens_helpers.py). Saved adapters exist under `data/training/runs/weighted_tokens_structure_smoke_01/language_model/`.

**Observed failure:** The saved generation repeats invented identity keys, includes unexpected appearance strings and becomes a long incomplete response. Its printed `Valid JSON` result is False.

**Lesson:** Training decision tokens more strongly did not make this saved generation structurally reliable. A teacher-forced training objective and free-running generation can behave differently. This observation does not isolate loss weighting as the sole cause.

## 10. Actual InternVL LoRA training: full assistant JSON loss

**Approach:** Keep the model/sample/adapter design and 100 smoke passes, but train all assistant JSON fields, structure and end tokens with weight 1.

**Sources:** [second configuration notebook](6_lora_training/TrainingWeightedTokens.ipynb), [named configurations](6_lora_training/README.md). Saved adapters exist under `data/training/runs/weighted_tokens_full_assistant_smoke_02/language_model/`.

**Observed result:** Saved epoch-100 weighted loss is about 0.00570. A displayed reload generation is labelled `Valid JSON: True` but `Exact smoke target: False`; it contains duplicate keys and unexpected IDs. The notebook also retains a NameError output, and that displayed reload refers to the backup project directory rather than the primary run. Do not present it as a clean controlled evaluation of the primary adapter.

**Lesson:** Low training loss and successful basic JSON parsing do not establish correct associations, unique keys, valid IDs, or even successful overfit generation. Duplicate-key detection and target/ID coverage matter. Held-out generalization is not demonstrated by this smoke run.

## 11. Component-specific adapter ablations

**Approach:** Implement frozen prompt/resolution/legend controls, a decoder overfit test, decoder q/v versus all attention/FFN targets, depth/rank changes, projector-only/decoder-plus-projector LoRA, and optional late-vision-block LoRA.

**Sources:** [full ablation guide](6_lora_training/README.md), [frozen controls](6_lora_training/02_frozen_baselines.ipynb), [decoder ablations](6_lora_training/04_decoder_lora_ablation.ipynb), [projector ablations](6_lora_training/05_projector_lora_ablation.ipynb), [optional vision ablations](6_lora_training/06_vision_lora_optional.ipynb).

**Outcome/lesson:** These individual ablations are implementations with completed execution unconfirmed. They distinguish reasoning/formatting adaptation from visual alignment and visual feature adaptation. They must not be described as all trained successfully, or as failed experiments without results.

## 12. Multipage preparation and reviewed page-wise training

**Approach:** Scale the modular detector/OCR/annotation process across a fixed sequence split, cache each page, review appearances/speakers, export randomized targets, and implement page-wise InternVL training/held-out validation.

**Sources:** [MultiPageTillAssociationPipeline guide](../MultiPageTillAssociationPipeline/README.md), [page annotation](../MultiPageTillAssociationPipeline/05_annotate_pages.ipynb), [training notebook](../MultiPageTillAssociationPipeline/6_lora_training/TrainingWeightedTokens.ipynb).

**Outcome/lesson:** Preparation outputs exist; this review found no `data/training/runs` directory in that project. Completed training is unconfirmed. Despite the project name, a training sample is one annotated page, not one joint three-page multimodal input. Sequence-level splitting and appearance-set validation prevent arbitrary ID numbering from becoming the evaluation criterion.

## 13. Independent character identity with DINO

**Approach:** Detect faces with the hysts YOLOv3 anime-face detector, encode crops with frozen DINOv2-small, and compare normalized 384-dimensional CLS vectors by cosine. Also implement face/head/upper-body crop variants and score-hidden human pair review.

**Sources:** [identity project guide](../MultiModelSystem/DINO-EXPERMINTATOIN-IDENTITY/README.md), [detector](../MultiModelSystem/DINO-EXPERMINTATOIN-IDENTITY/notebooks/02_character_detection.ipynb), [DINO pair test](../MultiModelSystem/DINO-EXPERMINTATOIN-IDENTITY/notebooks/03_dino_identity.ipynb), [human pair review](../MultiModelSystem/DINO-EXPERMINTATOIN-IDENTITY/notebooks/04_pair_review_and_score_analysis.ipynb).

**Outcome/lesson:** Detection/pair inspection outputs exist; the guide starts review from seven saved pairs. The exploratory 0.8 threshold is not a calibrated probability. Uncertain human labels and missed detections limit conclusions; the broader crop-strategy benchmark is not confirmed completed. Identity matching is separate from deciding which character spoke a bubble.

## 14. Separate whole-page OCR and learned Qwen attribution

**Approach:** Build a whole-page Paddle extraction pipeline with non-story exclusions and geometric line joining, then explore Qwen2.5-VL attribution using images/OCR and decoder attention/MLP LoRA.

**Sources:** [TextExtractionPipeline.ipynb](../MultiModelSystem/manga_notebook_codebase/notebooks/TextExtractionPipeline.ipynb), [joining validation](../MultiModelSystem/manga_notebook_codebase/notebooks/ocr_text_extraction/validation_ocr_joining.ipynb), [train_VL.ipynb](../MultiModelSystem/manga_notebook_codebase/notebooks/train_VL.ipynb).

**Observed training limitation:** The saved Qwen notebook reports 63 training sequences, 1,277 supervised groups and about 29.9 million trainable parameters; it also contains a CUDA OOM trying to allocate 2.35 GiB on the 8 GiB GPU. Adapter checkpoint directories exist, but that does not prove this interrupted run completed or generalizes.

**Lesson:** Separate OCR helps audit coverage and joining. LoRA still requires activation memory; a small trainable fraction does not guarantee the workload fits. Existing checkpoints and an OOM must be reported separately, not turned into a claimed final model win.

## 15. Independent panel detection and reading order

**Approach:** Use a YOLO26n panel detector and Manga109 panel-order helper, map cleaned bubbles to panels and approximate right-to-left bubble reading order.

**Sources:** [panel guide](7_panel_reading_order/README.md), [panel/order notebook](7_panel_reading_order/panel_reading_order.ipynb).

**Outcome/lesson:** Saved outputs exist. Unassigned bubbles/inseparable panel groups remain review cases. Correct detection/OCR can still make an incorrect transcript if reading order is wrong; this layout experiment does not retrain speaker association.

## How these approaches led into the Magi-based design

The earlier work explored both **end-to-end generative transcripts** and **a manually assembled modular pipeline**. It included separate bubble, face/body, identity, OCR and order systems; actual pretrained-model adapter training; and deliberately isolated component ablations.

The examples above exposed concrete difficulties: ID copying, repetition, malformed/incomplete output, weak correspondence between training loss and generated targets, manual supervision alignment, memory limits, and errors propagating between independently detected boxes. They motivate preserving explicit detections/links and auditing stages separately; they do not prove every earlier architecture is incapable of working.

Magi then provided a common source of text/character/panel/tail detections and learned links. The later system retained useful earlier concepts such as YOLO bubble candidates, Paddle OCR, sequence identity inspection, explicit reading-order handling and manual visual audits. Magi did not remove the need for artifact filtering, contour grouping or review of missing/wrong associations.

See the [complete 71-topic inventory](EXPERIMENT_INVENTORY.md) and [selected final-pipeline guide](../../Magi/paddleOCR-Magi/README_ARTIFACT_CONTOUR_FULL_INFERENCE.md). These 15 approach summaries reorganize existing topics; they are not 15 additional independent training runs. `1pageTest-EVERYTHING-GOOD` and `1pageTest-EVERYTHING-ULTRA-GOOD` are backup snapshots and are not counted twice.
