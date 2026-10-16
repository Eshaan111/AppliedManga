# InternVL3-2B LoRA training notebooks

These notebooks were created without executing cells, loading weights or starting training.
Use the **AppliedTaskByte** kernel and local base `C:/MODELS/InternVL3-2B`.

## Joint target maker

Run `00_make_assistant_targets.ipynb` after reviewing appearance groups and bubble speaker corrections in the local website. It exports the new joint characters/rejected_appearances/bubble_speakers JSON schema and a matching image with arbitrary B/A IDs. The notebook is unexecuted. The older training helpers below still use the earlier B-to-C schema and must be adapted before consuming these joint targets.

## Recommended order

| Order | Notebook | Purpose |
|---|---|---|
| 00 | review_labels_and_splits | Review speakers, duplicates, OCR; create grouped splits |
| 01 | inspect_parts_and_supervision | Inspect LoRA targets, assistant masking, image alignment |
| 02 | frozen_baselines | Remove numeric examples; compare image resolution, legend, text-only |
| 03 | decoder_lora_overfit_smoke | Fit reviewed smoke pages; prove gradient/save/reload path |
| 04 | decoder_lora_ablation | q/v → all attention → attention + FFN; then depth/rank |
| 05 | projector_lora_ablation | Projector only → decoder + projector |
| 06 | vision_lora_optional | Last-four vision blocks → decoder + vision; optionally all three |
| 07 | compare_reload_and_test | Select on validation, test once, export adapters |

The saved association output follows B1→C1 through B7→C7, then repeats C7 for B8–B15.
This is a failure pattern to investigate, not a source of training labels. A current-page
snapshot is supplied under `data/training/pages/page_smoke`; every label is null and reviewed
is false. Fill labels manually before 03. Review overlapping B4/B5 detections, and mark any
confirmed duplicate in exclude_bubbles. Also inspect B6/B8 and missing/off-panel speakers.

No ready human ground truth with local B/C IDs was supplied. Existing transcript labels in
other projects use a different character-ID scheme and are not silently converted. Every
included bubble needs a visible-instance C ID or UNKNOWN. Null means unreviewed, not UNKNOWN.

## Data

Each `data/training/pages/<page_id>/page.json` describes an annotated image snapshot, original
page dimensions, source hash, bubble and character boxes, OCR and human labels. See notebook 00
for snapshotting stage 1–4 outputs before the next page overwrites data/intermediate.
Use unique page_id values and a group_id per source book/chapter/sequence. Assign a complete
group to only one split: smoke, train, val or test. Page hashes cannot cross splits either.
At least separate reviewed train/val groups are required for 04–07; a separate test group is
required for the final test cell. One page can only establish that the model can overfit.

## Training design

Start with ordinary bf16/fp16 LoRA, rank 4 and decoder q/v in the final eight blocks.
Default batch size is one, accumulation four, one 448 px tile, three epochs and learning rate
1e-4. Smoke uses 40 passes and accumulation one. These are starting settings, not tuned claims.
All runs start from the same original local base and save adapters independently by component:
language_model, mlp1 and vision_model. Original weights stay frozen, including projector weights.
Projector experiments use LoRA on its two Linear layers. Vision experiments use LoRA on qkv/proj
in its final four blocks and are disabled by default. All-three is an optional experiment.

The helper uses the model's downloaded chat template and image preprocessing, checks image-token
counts, and supervises the assistant answer plus closing separator. It evaluates the output head
only at answer prediction positions to avoid materializing 151k-vocabulary logits for the whole
image/prompt prefix. Causal shifting is preserved. Prompt labels are masked; sequences are never
silently truncated. LoRA gradients are checked at the first optimizer step. The frozen image
branch uses no_grad, language checkpointing uses nonreentrant mode, and the custom vision branch
gets the gradient hook required by its upstream reentrant checkpoints.

Choose best epoch by validation loss, then compare generated ID accuracy in 07. Matching frozen
baselines prevent image settings from being mistaken for a training improvement. Dataset hashes,
settings, package versions, losses, gradient summaries, raw predictions and peak CUDA allocation
are saved under `data/training/runs`. Existing run names are never overwritten. Changing manifests
after training invalidates the comparison until data/results are reconciled. Reruns use new names.

## Memory and dependencies

Restart the kernel between notebooks if a model is still loaded. Keep only one model/kernel on
the GPU. The machine has 8 GiB VRAM; bf16, small rank, batch one and suffix loss reduce memory,
but actual training memory and runtime compatibility have not been tested. Keep sufficient free
disk for Windows' system-managed paging file; the earlier loading error was paging-file exhaustion.
On OOM, restart and reduce tiles/adapter depth. Full-model training and automatic quantization
changes are intentionally excluded. Tile count is an experiment setting: a full-page one-tile
image can lose visual detail, so inspect the frozen resolution comparison before broadening it.

The environment was observed to contain torch 2.14.0+cu132, transformers 4.52.4, peft 0.21.0 and
accelerate 1.15.0. No packages were installed. Do not reinstall the pipeline's generic requirements
wholesale into the shared environment. OCR still uses its separate OCR kernel only in stage 2.

References: [InternVL fine-tuning](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html),
[PEFT LoRA](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora).

## Decision-token training

TrainingWeightedTokens.ipynb is the alternative joint-JSON training path. It reads the exported sample.json and reuses the existing model loader, image preprocessing, LoRA configuration helper, and adapter saver. weighted_tokens_helpers.py provides load_joint_sample, build_assistant_with_spans, build_masked_target, collate_weighted_tokens, weighted_cross_entropy, and weighted_supervised_loss.

1stWeightConfig uses P/A/B IDs and UNKNOWN at weight 1.0; JSON structure (including empty lists) and the closing chat separator at weight 0.1; fixed field names and prompt/image/header at weight 0.0 with label -100. The loss is sum(weight * token_CE) / sum(weight). All assistant tokens remain in the input as context. Fast tokenizer offsets are checked against the existing model tokenizer IDs. A token overlapping an ID span is trained whole, including any inseparable punctuation. A token overlapping a fixed field name is ignored whole. Set formatting_weight and end_weight to zero for the earlier binary-mask baseline. Existing lora_helpers.py is unchanged.

Default: q/k/v/o LoRA in the final eight LLM layers, rank 4, alpha 8, dropout 0.05; vision/projector/base weights frozen; batch size one, one 448 px tile, learning rate 1e-4, 100 smoke passes. Set last_layers=None for all LLM layers. Inspect the token-mask cell before loading the model. The notebook was created without execution; model loading and training remain unverified. One-page fit is a smoke test, and generation checks include JSON validity.


## Named weight configurations

TrainingWeightedTokens.ipynb stores both configs in WEIGHT_CONFIGS. Change WEIGHT_CONFIG_NAME to select one.

- 1stWeightConfig preserves the previous run: decision_weight=1.0, formatting_weight=0.1, end_weight=0.1, field_weight=0.0. Its original run is weighted_tokens_structure_smoke_01.
- 2ndWeightConfig is selected by default: all four weights are 1.0. It trains on the entire assistant JSON plus EOS, while prompt/image/header tokens remain masked. Its new run is weighted_tokens_full_assistant_smoke_02.

Model, LoRA, sample and 100 epochs are identical. field_weight is configurable; its helper default remains 0.0 for compatibility with earlier configs. The old executed notebook and outputs are preserved verbatim in TrainingWeightedTokens_1stWeightConfig.ipynb. Saved runs/adapters are untouched.

Restart the kernel before the second experiment so it starts from the same base weights. Existing run folders cannot be overwritten; change the selected RUN_NAMES entry when repeating a config. The updated notebook was not executed.
