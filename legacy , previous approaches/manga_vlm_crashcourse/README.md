# Manga VLM Crash Course

A project-oriented crash course for building systems that read manga pages, recover story text in reading order, and maintain speaker identity across pages.

See [METRICS.md](METRICS.md) for definitions, formulas, limitations, and interpretation guidance for every training and generation metric.

The project goal is strong held-out results from the unmodified
[`TaskData/score.py`](TaskData/score.py). Both notebook workflows report its
`text_order_score` and `balanced_joint_f1` alongside the remaining official
metrics. Use these results to judge experiments; training loss and diagnostic
metrics help explain changes but do not replace the task scorer.

## Active pipeline

The active source of truth is `TaskData/development/labels.jsonl`. Each label
row maps directly to the naturally ordered images in
`TaskData/development/images/<sequence_id>/`.

Run [`notebooks-modelJSON/01_Baseline.ipynb`](notebooks-modelJSON/01_Baseline.ipynb) to validate
the canonical data and evaluate the untouched base model on the fixed eight
validation sequences. Then run
[`notebooks-modelJSON/02_Train_Evaluate.ipynb`](notebooks-modelJSON/02_Train_Evaluate.ipynb) to
train a fresh LoRA adapter and compare it on the exact same validation set.

The separate [ModelFormatter workflow](notebooks-ModelFormatter/README.md)
uses `images → VLM transcript → second text formatting model → task JSON`. Run its
baseline and trainer notebooks in order. Its prompt, transcript datasets,
adapters, and evaluation outputs are separate from the direct JSON experiment;
the canonical labels and fixed validation split are shared.

Earlier page-order augmentation artifacts are isolated under
`legacy/augmentation/` and are not part of current training or evaluation.

## Study order

0. Environment + sanity checks
1. Hugging Face VLM inference
2. How VLMs combine vision and language
3. Vision Transformer basics
4. LoRA / PEFT
5. VLM supervised fine-tuning
6. OCR pipelines
7. Bounding boxes and OpenCV
8. Manga reading order
9. Visual embeddings: CLIP / DINO
10. Character clustering and matching
11. Structured JSON generation
12. Evaluation and error analysis

## Recommended environment

Python 3.11 is a safe default for broad ML-library compatibility.

Create a venv:

```bash
python -m venv .venv
```

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Important: install PyTorch for your CUDA setup first

Use the official selector at https://pytorch.org/get-started/locally/ and install the exact command for your system.

Then:

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name manga-vlm --display-name "Manga VLM"
```

Install this project as an editable package so its Python modules can be
imported from scripts and notebooks:

```bash
python -m pip install -e .
```

In a notebook using the same `manga-vlm` kernel, imports then work from any
notebook directory:

```python
%load_ext autoreload
%autoreload 2

from Scripts.dataUtils import MangaSequenceDataset, load_labels
from Scripts.trainValSplit import fit_seq
```

## Suggested learning strategy

Run every notebook. Do not just read it.

For each notebook:
1. run the baseline code;
2. deliberately break one assumption;
3. inspect tensor shapes / intermediate outputs;
4. add one tiny experiment;
5. write 3-5 lines explaining what you learned.

## Manga-task decomposition

The task is easier to reason about as six subproblems:

1. transcription,
2. text filtering,
3. reading order,
4. speaker attribution,
5. cross-page identity consistency,
6. structured output and scoring.

A strong system does not have to solve all six with one model.

## Reproducibility

Keep:
- fixed random seeds,
- the fixed train/validation sequence IDs,
- config files,
- experiment names,
- dev predictions,
- scorer outputs,
- adapter checkpoints,
- logs of what changed between runs.

## Notes

Some OCR packages are optional and heavy. You do not need to install all OCR libraries on day one. Start with one path and add others when you have a reason.
