# Evaluation metrics

The active evaluator scores one naturally ordered prediction for each row in
`TaskData/development/labels.jsonl`. There are no augmented rows, page-order
variants, or permutation-specific metric sections.

`evaluate_model(...)` returns scores with nested `support`,
`unavailable_reasons`, `format_diagnostics`, and `generation_diagnostics`.
Unavailable scores are Python `None` / JSON `null`, displayed as **N/A**.
Zero means an evaluated score of zero, rather than an absent denominator.

## Notebook display and failure inspection

Both active notebooks use `display_metric_summary(...)`: format validity,
text error rate, dialogue F1, and speaker accuracy, each with evaluation
support. Other scores and counters are in a collapsed diagnostics section.
Targets and raw model responses remain visible for individual inspection.

`prediction_report(record)` and `inspect_prediction(...)` expose:

- Strict JSON/schema errors, including Markdown fences, flat dialogue lists,
  wrong field types, and page-count mismatches. Fenced content is inspected
  only to explain errors; it is never repaired or scored.
- Per-page text matches and similarities, unmatched target and predicted
  lines, raw speaker labels, and reading-order inversions. Item indexes are
  zero-based; page numbers are one-based. Character labels in these match
  records are raw; speaker accuracy uses the optimal label mapping.
- Missing/extra dialogue and page counts, UNKNOWN speaker counts, edit
  operation counts, and denominators for conditional scores.

`per_sequence_results.jsonl` saves these reports alongside raw predictions.
Review tables label valid-output CER separately from `failure_penalty_cer`,
the conservative invalid-output penalty. Baseline comparisons recompute
metrics from saved predictions so legacy zero defaults do not appear.

## Image delivery audit

Each canonical sample must contain exactly three images. The collator checks
that vision preprocessing returns all three and the chat template contains
three image placeholders. It then verifies Qwen's `image_grid_thw` has one row
per image, `pixel_values` contains all corresponding patches, and each input
row contains the expected number of image tokens after spatial merging.
Training and inference both stop with a clear error if these checks fail.

Evaluation repeats the tensor checks immediately before generation and saves
`image_paths` and `image_input_diagnostics` in `predictions.jsonl` and the
per-sequence report. The audit records ordered paths, per-image grid dimensions
and token counts, and total pixel patches. The notebooks show this audit for
the first sample. Older saved predictions have no image audit (`null`); their
image delivery cannot be retrospectively verified from the generated text.

## Primary goal

The authoritative benchmark is the unmodified `TaskData/score.py`. Both
workflows save its report as `task_score/score.json`, with a scorer-ready
submission and the exact held-out references. Prioritize its macro
`text_order_score` (page text accuracy and order) and `balanced_joint_f1`
(text plus speaker identity, balanced across identities). Also inspect
`joint_f1`, `matched_token_coverage`, and `speaker_accuracy_on_matched`.
Higher is better for all five. These scores use the task scorer's own
normalization, alignment, and speaker matching and differ from the diagnostic
metrics below. Invalid outputs are logged and omitted from the export; the
official scorer includes them as missing sequences, rather than excluding
them from the evaluation denominator.

The practical objective is maximum similarity between inference output and the
target. Read these together:

| Metric | Direction | Meaning |
| --- | --- | --- |
| `sequence_exact_match` | higher | Entire three-page output is correct after aligning arbitrary character IDs |
| `cer_valid_outputs` | lower | Character transcription error on structurally valid outputs |
| `cer_including_failures_as_deletions` | lower | Conservative CER that treats invalid outputs as missed text |
| `dialogue_text_match_f1` | higher | Correct coverage of target text items without extras, on valid outputs |
| `dialogue_f1_including_format_failures` | higher | End-to-end coverage, treating invalid outputs as missing all target lines |
| `matched_speaker_accuracy` | higher | Correct speaker on text-matched items |
| `b3_f1` | higher | Quality of character-identity grouping |
| `identity_switch_rate` | lower | Same character changing predicted identity across pages |
| `narration_f1` | higher | Correct narration identification |
| `json_parse_rate` | higher | Outputs that parse as JSON |
| `schema_compliance_rate` | higher | Outputs with the required page/item structure |

## CER and its components

Character Error Rate is:

```text
CER = (substitutions + deletions + insertions) / reference characters
```

- `cer_substitution_rate`: wrong characters replacing target characters.
- `cer_deletion_rate`: missing target characters, including completely missed
  text items.
- `cer_insertion_rate`: extra characters, including unmatched predicted items.

For valid outputs, these three rates sum to `cer_valid_outputs`, apart from
minor floating-point rounding.

`cer_valid_outputs` excludes predictions whose transcript schema cannot be
parsed. Always read it with parse/schema rates and `support`. With no valid
reference characters, it is N/A. The same applies to its component rates.

`cer_including_failures_as_deletions` counts all reference characters from an
invalid output as missing. It prevents malformed generations from making CER
look artificially good.

## Text-item matching

The evaluator normalizes harmless Unicode and whitespace differences, compares
text within the corresponding page, and makes a one-to-one assignment. A pair
must meet `minimum_similarity` (default `0.50`) to count as a match.

This is text-item matching, not spatial speech-bubble detection. The model does
not return bounding boxes.

`dialogue_text_match_f1` combines:

- true positives: matched target/predicted text items;
- false negatives: target items that were missed; and
- false positives: predicted items with no acceptable target match.

These counts use structurally valid responses only. If all responses are
invalid, dialogue F1 is N/A, not zero. `dialogue_precision` separates extra
lines from `dialogue_recall`, which separates missing lines. Precision is N/A
with no predicted lines; recall is N/A with no target lines. A valid empty
response against a nonempty target still has dialogue F1 = 0 and CER = 1.
Invalid target lines are counted as `support.unscored_target_dialogues`.
`dialogue_f1_including_format_failures` retains the end-to-end failure penalty.

## Reading order

`matched_dialogue_order_accuracy` checks the relative order of items that were
successfully text-matched. Because it ignores unmatched text, interpret it with
dialogue F1 and CER.

Reading order is N/A without matched dialogue pairs. Speaker accuracy is N/A
without matched lines. Identity switch rate is N/A without cross-page
transitions, and B-cubed F1 is N/A without matched character identities.

## Speaker and character identity

Names such as `character1` are local to each sequence. The evaluator therefore
finds the best one-to-one mapping between predicted and target character IDs
before computing speaker accuracy and aligned exact match.

- `matched_speaker_accuracy`: correctly attributed matched text items.
- `b3_f1`: clustering quality—same-character items should share an identity,
  and different-character items should remain separate.
- `identity_switch_rate`: cross-page transitions where the same target
  character changes predicted identity.

The mathematical speaker-label alignment is not data augmentation; it merely
avoids penalizing an otherwise equivalent renaming of character IDs.

## Narration

- `gt_narration_count`: narration items in the targets.
- `predicted_narration_count`: items predicted as `NARRATION`.
- `matched_narration_count`: text-matched items labeled `NARRATION` by both.
- `narration_f1`: balance of narration precision and recall.

Narration counts use valid outputs. F1 is N/A when no target or predicted
narration exists. The counts make a zero F1 interpretable. For example, ground-truth narration
with zero predicted narration is a genuine omission rather than a matching
ambiguity.

## Structure

- `json_parse_rate`: fraction of raw outputs that parse as JSON.
- `schema_compliance_rate`: fraction matching the expected nested page lists,
  where every item contains exactly string-valued `speaker` and `text` fields.
- `page_count_match_rate`: fraction of structurally valid responses with one
  page list per target image. Missing/extra pages still receive content
  penalties and are flagged separately from JSON/schema errors.

The expected response is:

```json
[
  [{"speaker": "character1", "text": "..."}],
  [],
  [{"speaker": "NARRATION", "text": "..."}]
]
```

## Exact match

`sequence_exact_match` requires every page, item, text, order, and speaker to be
correct after optimal character-ID alignment.

`literal_sequence_exact_match` additionally requires literal character labels
to match. It is stricter and mainly diagnostic.

## Generation diagnostics

The nested `generation_diagnostics` object reports:

- `generated_sequences`;
- `hit_max_new_tokens`;
- `suspected_repetition_loops`;
- minimum generated tokens;
- maximum generated tokens; and
- mean generated tokens.

Max-token hits and repetition loops should be inspected manually because they
can cause invalid JSON, hallucinated text, and high insertion rates.

## Recommended reading order

1. Confirm JSON parse and schema compliance are high.
2. Check conservative CER and dialogue F1.
3. Inspect substitution, deletion, and insertion rates.
4. Check narration counts and narration F1.
5. Check speaker accuracy, B-cubed F1, and identity switches.
6. Inspect sequence exact match and the worst individual predictions.
7. Inspect any max-token or repetition-loop generations.
