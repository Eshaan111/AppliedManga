Yep. Here is the whole workflow, cleaned up with the choices we just made.

## 1. Unit of training = one full 3-page sequence

Each sample is:

```text
Sequence
├── Page 1
├── Page 2
└── Page 3
```

and contains **all Magi-detected text boxes** from those pages.

Do not split boxes into independent training samples.

For each box keep:

```python
{
    "box_id": "PG2:T4",
    "page_index": 2,
    "ocr": "...",
    "bbox": [x1, y1, x2, y2],

    "magi_visual_embedding": ...,

    "tail_score": ...,

    "label": "SPOKEN" | "NARRATED" | -100
}
```

We are removing `OTHER` for now.

So:

```text
characterN → SPOKEN
NARRATION  → NARRATED
UNKNOWN    → ignore
unmatched  → ignore
```

---

## 2. Train/validation split

Split by **sequence**, never by box or page.

Example:

```text
65 train sequences
15 validation sequences
```

All class statistics are computed only from the training split.

---

## 3. Sequence-level oversampling

We do not duplicate anything inside a sequence.

Instead, narration-containing sequences can be sampled more often.

For example:

```text
Sequence A: contains narration → sampling weight 2
Sequence B: no narration       → sampling weight 1
```

When A is sampled again, it is still the exact same intact sequence:

```text
same 3 pages
same boxes
same context
```

This addresses the fact that relatively few sequences may contain narration.

---

# 4. Semantic feature: DeBERTa

For every target text box, use:

```text
previous OCR on the same page
+
target OCR
```

For example:

```text
PG1:T1: "..."
PG1:T2: "..."
TARGET PG1:T3: "..."
```

I would initially use maybe the previous 2 boxes on the same page:

```text
previous_2 [SEP]
previous_1 [SEP]
TARGET
```

We do **not** need future text initially if we want to preserve a natural contextual direction.

DeBERTa produces:

\[
t_i \in \mathbb{R}^{768}
\]

Initially:

```text
DeBERTa = frozen
```

This gives us semantic information about:

> what this text means in its local discourse context.

---

# 5. Visual feature: Magi

For the same text box, use the Magi text-object representation:

\[
v_i = \text{Magi text token}
\]

Magi stays frozen.

So we do not train another visual encoder.

This captures the visual nature of the region and page context.

---

# 6. Geometry

For every box:

\[
g_i =
[
x_c/W,\;
y_c/H,\;
w/W,\;
h/H
]
\]

plus page information separately.

Optionally:

```text
aspect ratio
normalized area
```

can be added later.

But start small.

---

# 7. Tail information

Use Magi's tail evidence.

Prefer something like:

```python
tail_features = [
    max_tail_score,
    has_tail
]
```

rather than only a boolean.

So the model can distinguish:

```text
tail score = 0.03
```

from:

```text
tail score = 0.94
```

This is a very strong speech cue.

---

# 8. Project each modality

Example:

```text
DeBERTa: 768 → 256
Magi:    D   → 256
Geometry:4   → 32
Tail:    2   → 16
```

Then concatenate:

\[
z_i =
[t'_i;v'_i;g'_i;r'_i]
\]

and project again:

```text
~560 dimensions
↓
Linear
↓
256 dimensions
```

So every text box becomes one common representation:

\[
z_i \in \mathbb{R}^{256}
\]

---

# 9. Positional information

Add:

```text
sequence-position embedding
+
page embedding
```

So:

\[
z'_i
=
z_i
+
e_{\text{position}}
+
e_{\text{page}}
\]

No RoPE initially.

We already have:
- sequence position
- page identity
- actual box geometry

which should be enough for V1.

---

# 10. Build one sequence of box tokens

Suppose:

```text
Page 1: 8 boxes
Page 2: 5 boxes
Page 3: 7 boxes
```

Then the model gets:

\[
Z \in \mathbb{R}^{20 \times 256}
\]

ordered:

```text
PG1:T1
PG1:T2
...
PG2:T1
...
PG3:T7
```

Use your existing box order.

---

# 11. Sequence Transformer

Feed the complete sequence through a small Transformer:

```text
d_model = 256
heads = 4
layers = 2
dropout = 0.1
```

So:

```text
all box representations
↓
Transformer
↓
contextualized box representations
```

Each box can now use information from the whole 3-page sequence.

This is where the system gets broader context like:

```text
several previous boxes are clearly a character's internal monologue
```

or:

```text
this region looks and reads like narrator exposition
```

---

# 12. Classification head

For every box:

```python
Linear(256, 2)
```

Output:

```text
SPOKEN
NARRATED
```

So for \(N\) boxes:

\[
\text{logits} \in \mathbb{R}^{N\times2}
\]

---

# 13. Attention mask

Different sequences have different numbers of boxes.

Example:

```text
Sequence A = 14 boxes
Sequence B = 22 boxes
```

Batching pads A to 22.

So:

```text
real box → attention = 1
padding  → attention = 0
```

This mask is only for the sequence Transformer.

It answers:

> Is this a real box or just padding?

---

# 14. Loss masking

Separate issue.

Some real boxes have no reliable GT:

```text
unmatched
UNKNOWN
```

They still remain visible to the Transformer as context.

But:

```text
label = -100
```

so they do not contribute to the supervised loss.

So:

```text
real + labeled:
attention = yes
loss = yes

real + unlabeled:
attention = yes
loss = no

padding:
attention = no
loss = no
```

That distinction is important.

---

# 15. Class-Balanced Cross-Entropy

Now count training labels globally:

```text
SPOKEN   = n_s
NARRATED = n_n
```

Use effective-number weighting:

\[
w_c \propto
\frac{1-\beta}
{1-\beta^{n_c}}
\]

Then normalize the weights.

So if narration is much rarer:

```text
w_narrated > w_spoken
```

For each labeled box:

\[
L_i
=
-w_{y_i}\log P(y_i)
\]

Unlabeled boxes are ignored.

---

# 16. Sequence loss

Keep it simple initially:

\[
L_{\text{seq}}
=
\frac{1}{N_{\text{labeled}}}
\sum_i L_i
\]

No additional per-sequence balancing at first.

We already have two imbalance mechanisms:

```text
1. narration-containing sequence oversampling
2. global Class-Balanced CE
```

Do not stack more tricks until we know they are needed.

---

# 17. Backpropagation

Initially freeze:

```text
DeBERTa
Magi
```

Train only:

```text
text projection
visual projection
geometry projection
tail projection
fusion layer
sequence Transformer
classifier
```

This keeps the trainable model small.

Later:

```text
semantic mistakes remain high
→ unfreeze final DeBERTa layer

visual mistakes remain high
→ investigate Magi-side adaptation
```

---

# 18. What to log

Per epoch:

```text
train loss
val loss

SPOKEN precision / recall / F1
NARRATED precision / recall / F1

macro F1
balanced accuracy
```

Also per class:

```text
mean p_true
median p_true

fraction with p_true > 0.9
fraction with p_true < 0.5
```

And gradient norms:

```text
fusion grad norm
Transformer grad norm
classifier grad norm
```

This lets us later determine whether **Class-Balanced Focal Loss** is warranted.

If:

```text
most SPOKEN examples are already trivial
but still contribute lots of gradient
```

then focal loss becomes a good next experiment.

---

# 19. Save every validation prediction

For every labeled box:

```text
sequence_id
box_id
OCR
GT
prediction
p_spoken
p_narrated
tail_score
bbox
page
```

This lets us isolate exactly:

```text
SPOKEN → NARRATED
```

especially internal thought mistakes.

And:

```text
NARRATED → SPOKEN
```

for narrator captions.

---

# Full architecture

```text
                    FULL 3-PAGE SEQUENCE

                       every Magi box
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
       OCR              Magi visual            Geometry
        │                  token                + tail
        │                    │                     │
 previous page-local OCR     │                     │
        │                    │                     │
     DeBERTa                 │                     │
        │                    │                     │
        └──────────── projections ─────────────────┘
                             │
                           fusion
                             │
                    one 256-d token / box
                             │
               + position + page embeddings
                             │
               FULL 3-PAGE BOX SEQUENCE
                             │
                 2-layer Transformer
                             │
                   contextual box vectors
                             │
                     Linear(256, 2)
                             │
                  SPOKEN / NARRATED
                             │
              Class-Balanced Cross-Entropy
                             │
                         backward
```

And around the entire training loop:

```text
narration-containing sequences
sampled somewhat more frequently
```

So the core design is now:

\[
\boxed{
\text{DeBERTa semantics}
+
\text{Magi visual}
+
\text{geometry}
+
\text{tail evidence}
+
\text{3-page Transformer context}
\rightarrow
\text{SPOKEN/NARRATED}
}
\]

with **whole-sequence oversampling + Class-Balanced Cross-Entropy** handling the narration imbalance.