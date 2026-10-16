# Detected failures

## F001 — Baseline prediction loses page structure

- **Detected:** 2026-10-01
- **Notebook:** [01_Baseline.ipynb](01_Baseline.ipynb), `single-sample-inference` cell
- **Sequence:** `seq_d8bc3c6ac1bb6017` (`SAMPLE_INDEX = 0`)
- **Status:** Observed; unresolved.
- **Expected:** A JSON array containing three inner lists, one per input page, with dialogue objects containing `speaker` and `text`.
- **Observed:** The saved raw base-model prediction contains a flat list of four dialogue objects, wrapped in a Markdown JSON code fence. It does not preserve the three page lists.
- **Impact:** The saved single-sample metrics report `json_parse_rate = 0`, `schema_compliance_rate = 0`, and `cer_evaluated_sequences = 0`. The reported `cer = 0` therefore does not indicate correct transcription; `cer_including_failures_as_deletions = 1` records the failure.
- **Input check:** The prompt builder supplies all page images from the selected sequence. This is an output-format failure, not evidence of single-page input.

## F002 — Baseline generation reaches the token limit

- **Detected:** 2026-10-01
- **Notebook:** [01_Baseline.ipynb](01_Baseline.ipynb), `run-baseline` cell
- **Evidence:** [predictions.jsonl](../outputs/canonical_base_baseline/predictions.jsonl) and [metrics.json](../outputs/canonical_base_baseline/metrics.json).
- **Status:** Observed; unresolved.
- **Observed:** Four of eight sequences generate 1,024 tokens without reaching EOS: `seq_2032620aa4e4ac7f`, `seq_535e0754d805105b`, `seq_dce8be23c3281228`, and `seq_098e260f231aa789`. Saved predictions include repeated dialogue with increasing character labels; the existing token-pattern detector flags zero suspected repetition loops.
- **Impact:** Outputs are cut off at the generation limit. Evaluation generates 5,170 tokens in total, compared with 98 for the preview, so extrapolating eight times the preview runtime substantially underestimates generation work. No per-sequence timing is saved, so the exact runtime contribution is unmeasured.
