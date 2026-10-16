# How I read the scores

I checked [TaskData/score.py](TaskData/score.py), version 2.1, to understand what each number actually measures. This guide follows that code. There are five separate scores, not one combined total.

The useful distinction for this project is simple: recovering the text and identifying the speaker are different jobs. A line can have perfect OCR and the right character crop, then still get a wrong speaker ID because clustering split that person into several entities. I explain that effect below. The [main README](README.md) has the pipeline story; the [fine-tuning guide](FINETUNING-EXPERMENTS/README.md) covers what I tried to improve it.

## Input and normalization

One JSONL line must contain exactly sequence_id and pages. The default task requires exactly three ordered page lists. Each transcript entry must contain exactly speaker and text. Entries and sequence IDs must satisfy the loader's nonempty/trimmed-string checks; an empty page is []. Duplicate keys and duplicate sequence IDs are rejected.

Text is NFKC-normalized and case-folded. The scorer standardizes listed curly quotes, ellipsis and dash variants, removes Unicode format characters, collapses whitespace, standardizes repeated dots and removes spaces around selected punctuation. It does not remove all punctuation. Words, numbers and punctuation are tokenized; word-internal apostrophes/hyphens can remain inside a token.

A reference UNKNOWN speaker becomes an unknown identity (None). Its text still contributes to text metrics. A predicted UNKNOWN is an ordinary anonymous predicted speaker, not an ignored prediction.

## Token matching and alignment

For two normalized tokens a and b, the matching weight is:

```text
w(a,b) = 1                                      if a == b
         0                                      if unequal and min(len(a),len(b)) < 3
         1 - Levenshtein(a,b)/max(len(a),len(b))  if that value >= 0.6
         0                                      otherwise
```

The scorer finds a maximum-total-weight monotonic alignment separately on each page. It cannot reorder tokens to rescue the prediction. Entry boundaries are flattened for alignment; splitting/concatenating entries is not itself scored, provided text order and token speaker labels remain correct.

The same alignment is used for speaker scoring. Anonymous character labels are matched one-to-one across the entire sequence using a Hungarian assignment. Ordinary joint metrics optimize correctly attributed token weight; balanced joint F1 computes a separate mapping optimizing the balanced objective. NARRATION is fixed to NARRATION and cannot be renamed to a character. A predicted identity cannot map to several reference people.

## 1. text_order_score

For each page, join all entry texts in their existing order, then normalize. Let g and p be those normalized page strings:

```text
page_score = max(0, 1 - Levenshtein(g,p)/max(len(g),len(p),1))
sequence_score = mean(page_score over pages where g or p is nonempty)
```

If all pages are empty, the sequence-level score is 1. A page empty on both sides is otherwise excluded from the sequence mean. This is a character-edit similarity over the ordered page text, not a dedicated geometric reading-order metric or exact text accuracy percentage.

The final report averages sequence scores. Sequences without active text pages are excluded from this macro average when active or missing sequences exist; otherwise the scorer falls back to all sequences. A missing sequence is explicitly assigned zero text order and zero coverage.

Current macro result: 0.9022580406591174.

## 2. matched_token_coverage

Let G_all be the number of reference tokens, including tokens whose reference speaker is UNKNOWN. Let W_all be the sum of matching weights over all aligned token pairs:

```text
sequence_coverage = W_all / G_all
```

If G_all is zero, the result is 1 when there are no predicted tokens, and 0 otherwise. Macro averaging uses the same active/missing-sequence eligibility convention as text order.

This is weighted reference-token recovery. Approximate matches can receive fractional credit. Extra predicted tokens do not directly enlarge its denominator; they are penalized by other metrics. Coverage is not precision.

Current macro result: 0.9239508442739283.

## The quantities used in the speaker scores

Within one sequence:

```text
G = number of reference tokens with a known speaker, including NARRATION
P = number of predicted tokens, minus predictions aligned to reference UNKNOWN tokens
A = sum of alignment weights on pairs whose reference speaker is known
C = sum of those weights where the ordinary speaker mapping gives the correct speaker
```

The UNKNOWN masking in P subtracts one token for each aligned unknown-reference pair, not its fractional alignment weight. Unmatched predicted tokens still count. Identities and their token counts span all three pages.

## 3. speaker_accuracy_on_matched

```text
sequence_speaker_accuracy = C / A
```

This asks whether already-aligned known-speaker text has the correct identity after the ordinary mapping. It does not directly penalize missing or extra unaligned tokens. If A is zero but G is positive, the score is 0. If G is zero, the score is None and the sequence is omitted from the speaker-metric macro average.

Current macro result: 0.6389564706543975. This is a mean of sequence ratios, not the global sum(C)/sum(A).

## 4. joint_f1

```text
joint_precision = C / P
joint_recall = C / G
sequence_joint_f1 = 2*C / (G+P)
```

The final formula is the implemented one. Missing reference text, extra predictions and wrong speakers all hurt it. Correctly attributed tokens from a talkative identity can outweigh errors on short identities within a sequence. When G is zero, this metric is None.

The report averages valid sequence-level values; it is not one dataset-wide micro-F1.

Current macro result: 0.6004168626206221.

## 5. balanced_joint_f1

For known reference identity g and predicted identity p, define:

```text
G_g = all known reference tokens belonging to g
P_p = all scored predicted tokens belonging to p, after reference-UNKNOWN masking
W_gp = total aligned matching weight for token pairs labelled g and p
F_gp = 2*W_gp / (G_g + P_p)
```

The scorer finds the one-to-one identity mapping maximizing the sum of F_gp contributions, with NARRATION fixed. This mapping can differ from the mapping used for ordinary joint F1.

Let K be the number of known reference identities represented by tokens, including NARRATION when present. Let E be the number of scored predicted identities whose balanced mapping is absent or does not name an existing reference identity. These surplus identities include unmatched character labels, UNKNOWN and false NARRATION labels.

```text
sequence_balanced_joint_f1 = min(1, sum(F_gp for correctly mapped identity pairs)/(K+E))
```

A reference identity with no correctly mapped prediction contributes zero but still counts in K. A surplus predicted identity contributes zero and increases E. When G is zero, the sequence's balanced score is None and is omitted from the macro average.

The final report is the mean of eligible sequence scores. It balances identities within each sequence, penalizes surplus identities, then balances sequences in the outer average. It is not a conventional two-class SPOKEN/NARRATED macro-F1 and is not just ordinary joint F1 with different speaker names.

Current macro result: 0.3904260803788208.

## Why joint and balanced joint can differ so much

Suppose the reference has three speakers contributing 100, 10 and 10 tokens. All 120 tokens are recovered perfectly, but every token is assigned to one predicted speaker.

```text
Text coverage = 1
Ordinary mapping credits the 100-token speaker:
Joint F1 = 2*100/(120+120) = 0.833333

That speaker's identity F1 = 2*100/(100+120) = 0.909091
Other two reference identities contribute zero:
Balanced joint F1 = 0.909091/3 = 0.303030
```

Adding surplus identities can reduce balanced F1 further because its denominator becomes K+E. Conversely, merely renaming characters consistently has no penalty: the scorer already optimizes the identity mapping.

## Why clustering can hurt even with the right character crop

**Yes, inconsistent clustering directly hurts balanced joint F1.** The same person's front view and side view might both receive the correct text links, but end up as two predicted entities. Only one of those entities can map to that reference person. The other loses speaker credit and counts as an extra identity.

I checked this with score.py using a small constructed example, not a new experiment result: one reference speaker says eight perfectly recovered tokens, split equally between two predicted speaker IDs.

| Metric | One consistent identity | Same person split into two IDs |
| --- | ---: | ---: |
| Text order | 1.0 | 1.0 |
| Matched token coverage | 1.0 | 1.0 |
| Speaker accuracy | 1.0 | 0.5 |
| Joint F1 | 1.0 | 0.5 |
| Balanced joint F1 | 1.0 | 0.333333 |

For the split case, the chosen identity gets `2*4/(8+4) = 2/3` identity F1. There is one reference identity and one extra prediction, so balanced joint F1 is `(2/3)/(1+1) = 1/3`.

Merging different people into one predicted entity also hurts: that entity can match only one reference speaker. Simply renaming a character consistently is fine because the scorer finds the best mapping.

**Clustering was the main problem I noticed when looking through the outputs**, and it is what I would work on first. This calculation explains how those mistakes can compound even when the original association is right. It does not establish that every attribution error came from clustering; original text-character links can also be wrong.

## What I take from the development scores

All 80 sequences are present. The speaker metrics average over 79 eligible sequences; one reference sequence has no known speaker tokens.

The analysis found 272 known sequence-local reference identities, 29 with zero identity F1, and 170 surplus predicted identities across the speaker-scored sequences. Those surplus labels consist of 102 character labels, 47 NARRATION labels and 21 UNKNOWN labels after the scorer's masking rules. These are dataset totals; they do not form one global balanced-F1 denominator because each sequence is scored separately.

So I recovered most of the text, but still struggled to give it consistent speaker identities. Approximately 72.7% of the wrongly attributed aligned-token weight is wrong spoken identity, 24.7% is speech called narration, and 2.5% is narration called speech. These percentages describe the combined wrongly attributed token weight, not the way the final sequence averages are calculated. The [per-identity table](submission_outputs/inference_development/analysis/per_identity_f1.csv) has the counts and F1 values.

The saved analysis checks the four token/speaker scores against the official functions. It reads text order from the saved report instead of recalculating character edit distance. I have not changed score.py, the predictions or score.json while editing these guides.

## What I would check next

I would inspect identity grouping separately from text-character links, then check the final transcript again. That should show whether a change fixes the right problem rather than just moving mistakes around. For learned narration models, I would also report narration precision, recall, F1 and support: those are different from this scorer's identity-balanced joint F1.

The [model links](README.md#model-links) and [next experiments](README.md#what-i-would-try-next) are in the main README.
