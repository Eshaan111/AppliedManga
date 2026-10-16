# Manga VLM Strategy Tuning

## Current Evaluation Snapshot

| Area                 |                 Metric |             Result | Interpretation                 |
| -------------------- | ---------------------: | -----------------: | ------------------------------ |
| Structure            |        JSON Parse Rate |    **91.7%** | Strong                         |
| Structure            |      Schema Compliance |    **91.7%** | Strong                         |
| Ordering             | Reading Order Accuracy |    **90.5%** | Strong                         |
| Dialogue Extraction  |            Dialogue F1 |    **71.9%** | Moderate                       |
| Transcription        |                    CER | **53.9% ↓** | Major weakness                 |
| Speaker Attribution  |       Speaker Accuracy |    **49.9%** | Major weakness                 |
| Identity Consistency |                 B³ F1 |    **57.7%** | Weak / moderate                |
| Identity Consistency |   Identity Switch Rate | **75.4% ↓** | Major weakness                 |
| Narration            |           Narration F1 |       **0%** | Complete failure mode          |
| End-to-End           |   Sequence Exact Match |       **0%** | Expected given upstream errors |

---

# Main Diagnosis

The model is **not primarily failing at reading order**.

The strongest parts of the current system are:

- JSON formatting
- Schema compliance
- Reading order

The weakest parts are:

1. **Transcription quality**
2. **Character identity consistency**
3. **Speaker attribution**
4. **Narration recognition**

This means architectural changes should target those weaknesses rather than adding complexity to areas that already perform well.

---

# Architecture Decisions

## Do NOT Add a Reading-Order Head Yet

Current reading-order accuracy:

```text
90.5%
```

This is already one of the strongest metrics.

An order-classification head would currently solve a problem that is not a major bottleneck.

### Decision

```text
Reading-order head: NO
```

Only reconsider if reading-order accuracy drops substantially on a larger and more representative validation set.

---

## Do NOT Add a Fixed Character-Classification Head

A head such as:

```text
representation
    ↓
Linear
    ↓
CHARACTER_1 / CHARACTER_2 / CHARACTER_3
```

is a poor fit because character labels are sequence-relative.

For example:

```text
Sequence A:
CHARACTER_1 = John

Sequence B:
CHARACTER_1 = Batman
```

`CHARACTER_1` does not represent a global semantic class.

### Decision

```text
Fixed character classifier: NO
```

---

## Identity / Contrastive Head Is Worth Investigating

Current identity-related results:

```text
Speaker Accuracy      ≈ 49.9%
B³ F1                 ≈ 57.7%
Identity Switch Rate  ≈ 75.4%
```

These strongly suggest that maintaining character identity across pages is a major weakness.

Instead of predicting global character classes, train representations such that:

```text
same character
      ↓
similar embeddings

different characters
      ↓
dissimilar embeddings
```

Possible objectives:

- Supervised contrastive loss
- Contrastive loss
- Triplet loss

Potential future objective:

```text
L_total = L_SFT + λ * L_identity
```

### Decision

```text
Identity / contrastive auxiliary objective: PROMISING
```

However, do not add it immediately until transcription and evaluator reliability have been checked.

---

# Priority 1 — Verify the Evaluator

Before changing the model, verify that the metrics themselves are behaving correctly.

Particularly inspect:

```text
Narration F1 = 0
Identity Switch Rate
Dialogue matching
Speaker matching
```

## Narration Debugging

Record:

```python
gt_narration_count
predicted_narration_count
matched_narration_count
```

Interpretation:

```text
GT = 15
Pred = 0
```

means the model has not learned narration.

But:

```text
GT = 15
Pred = 12
TP = 0
```

could indicate a matching/evaluation issue.

Also check label normalization such as:

```text
NARRATION
NARRATOR
narration
```

---

# Priority 2 — Improve Transcription

Current:

```text
CER ≈ 0.539
```

This is high.

Roughly speaking, this corresponds to about 54 character-level edit operations per 100 reference characters.

Before attributing everything to speaker identity, determine what kind of transcription errors dominate.

Break CER into:

```text
Substitution Rate
Deletion Rate
Insertion Rate
```

## Interpretation

### Mostly deletions

Likely:

- missed bubbles
- unread text
- incomplete generation

### Mostly substitutions

Likely:

- OCR / visual recognition problem
- difficult fonts
- small text
- weak visual features

### Mostly insertions

Likely:

- hallucination
- duplicated dialogue
- repetition loops
- over-generation

This decomposition should guide further tuning.

---

# Priority 3 — Character Identity Consistency

After transcription becomes reasonably reliable, explicitly address character consistency.

The current metrics justify experimenting with:

```text
Qwen2.5-VL
    +
LoRA SFT
    +
Identity projection head
    +
Contrastive objective
```

Conceptual architecture:

```text
          Qwen / Vision representations
                     │
               ┌─────┴─────┐
               │           │
          LM Generation   Identity Projection
               │           │
               ▼           ▼
         Transcript     Embeddings
               │           │
           CE Loss   Contrastive Loss
```

Total loss:

```text
L_total = L_SFT + λ * L_identity
```

Compare this against the existing baseline.

Primary metrics to inspect:

```text
B³ F1
Speaker Accuracy
Identity Switch Rate
```

The goal is to determine whether explicit identity supervision provides measurable benefit.

---

# Priority 4 — Narration

Current:

```text
Narration F1 = 0
```

First verify the evaluator.

If the failure is genuine, try the simplest interventions first:

1. Improve prompt instructions
2. Ensure narration examples are represented sufficiently in training
3. Normalize output labels
4. Examine narration-vs-dialogue visual patterns
5. Rebalance examples if narration is rare

Only consider a dedicated narration classification head if the simpler approaches fail.

Potential auxiliary objective:

```text
L_narration = BCE(predicted_is_narration, target_is_narration)
```

---

# Page-Reordering Experiment

Results:

## Canonical

```text
CER               0.401
Dialogue F1       0.821
Order Accuracy    0.905
Speaker Accuracy  0.500
B³ F1             0.531
```

## Reordered

```text
CER               0.570
Dialogue F1       0.697
Order Accuracy    0.905
Speaker Accuracy  0.499
B³ F1             0.587
```

The reordered sequences show substantially worse:

```text
CER:         0.40 → 0.57
Dialogue F1: 0.82 → 0.70
```

while reading-order performance remains almost unchanged.

This suggests that chronological page context may help the VLM understand the sequence.

### Current Recommendation

Do **not** freely permute page order as data augmentation unless arbitrary page order is part of the real test distribution.

However, this conclusion is preliminary because:

```text
canonical sequences = 4
reordered examples  = 20
```

and the dataset contains only:

```text
independent_sequence_groups = 4
```

The permutations are correlated versions of the same original data.

Therefore the effective independent sample size is much closer to **4**, not 24.

---

# Generation Diagnostics

Current diagnostics:

```text
Generated sequences:          24
Hit max_new_tokens:            2
Suspected repetition loops:    2
Min generated tokens:        329
Max generated tokens:       1024
Mean generated tokens:       641
```

The generations reaching:

```text
1024 tokens
```

deserve manual inspection.

They may contribute disproportionately to:

- invalid JSON
- dialogue hallucinations
- high CER
- zero exact-match rate
- duplicate text

Add a debug inspection such as:

```python
if hit_max_new_tokens or repetition_loop:
    print(sequence_id)
    print(output)
```

Also inspect whether stopping criteria or end-of-response behavior are working correctly.

---

# Recommended Experimental Order

## Experiment 1 — Baseline Validation Audit

Keep the existing architecture.

Verify:

```text
CER calculation
Dialogue matching
Speaker alignment
Narration matching
Identity switch calculation
```

Do not modify the model yet.

---

## Experiment 2 — CER Error Decomposition

Measure:

```text
Insertion rate
Deletion rate
Substitution rate
```

Then manually inspect representative examples from each failure category.

Goal:

```text
Determine whether transcription failure comes from OCR,
missed text, hallucination, or generation behavior.
```

---

## Experiment 3 — Generation Stability

Investigate:

```text
max-token hits
repetition loops
over-generation
invalid outputs
```

Potential tuning dimensions:

- max_new_tokens
- repetition penalty
- stopping criteria
- prompt format
- generation temperature
- EOS behavior

---

## Experiment 4 — Improve SFT Baseline

Improve the existing LoRA + SFT model before adding heads.

Evaluate primarily on:

```text
CER
Dialogue F1
Speaker Accuracy
B³ F1
Identity Switch Rate
```

---

## Experiment 5 — Identity Auxiliary Objective

Once the baseline is trustworthy:

```text
Baseline:
Qwen2.5-VL + LoRA + SFT
```

vs.

```text
Experiment:
Qwen2.5-VL + LoRA + SFT + Identity Contrastive Loss
```

Compare:

```text
B³ F1
Speaker Accuracy
Identity Switch Rate
CER
```

The last metric matters because improving identity should not substantially damage transcription.

---

# Metrics to Prioritize

## Headline Metrics

```text
CER ↓
Dialogue F1 ↑
Speaker Accuracy ↑
B³ F1 ↑
Sequence Exact Match ↑
```

## Diagnostic Metrics

```text
Reading Order Accuracy ↑
Identity Switch Rate ↓
Narration F1 ↑
JSON Parse Rate ↑
Schema Compliance Rate ↑
Insertion / Deletion / Substitution Rates ↓
```

---

# Current Strategy Summary

```text
                     CURRENT MODEL

JSON formatting         █████████░  Strong
Reading order           █████████░  Strong

Dialogue detection      ███████░░░  Moderate

Identity clustering     ██████░░░░  Weak
Speaker attribution     █████░░░░░  Weak

Transcription           ████░░░░░░  Major issue
Narration               ░░░░░░░░░░  Failed
```

## Current Decisions

```text
Reading-order head            → NO
Fixed character classifier    → NO
Identity contrastive head     → POSSIBLY, AFTER BASELINE DEBUGGING
Narration head                → ONLY IF SIMPLE FIXES FAIL
More architectural complexity → NOT YET
```

The central strategy is:

```text
Measure failure
    ↓
Verify the metric
    ↓
Identify the actual bottleneck
    ↓
Run the smallest targeted intervention
    ↓
Re-evaluate
    ↓
Only then modify architecture
```

The immediate focus should therefore be:

```text
1. Evaluator verification
2. CER error decomposition
3. Generation stability
4. Better SFT baseline
5. Identity-specific auxiliary objective
```
