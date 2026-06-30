# ARC-CLIP

A CLIP-style contrastive model that aligns **visual transformations** (ARC
demonstration pairs) with **symbolic DSL programs** in a shared embedding space,
used as a learned heuristic for program retrieval on the Abstraction and
Reasoning Corpus (ARC-AGI).

Two encoders are trained with a symmetric InfoNCE loss so that a task's
demonstrations and its ground-truth program land at the same point in space:

- **phi_v** — a CNN over demonstration pairs.
- **phi_p** — a Transformer over arc-dsl program tokens.

At inference, the demonstrations of an unseen task are encoded and used to rank
candidate programs by cosine similarity, turning blind DSL search into a guided,
verifiable search.

## Data

- [`arc-dsl`](https://github.com/michaelhodel/arc-dsl) — a verified DSL program
  for each of the 400 ARC training tasks (160 primitives).
- [`re-arc`](https://github.com/michaelhodel/re-arc) — 1000 procedurally
  generated input/output pairs per task.

Together these give labeled `(demonstrations, program)` pairs at scale.

## Method (Objective 1, Part A)

The 400 tasks are split into **train (~320)** and a **held-out test (~80)** set.
ARC-CLIP is trained only on the train tasks. We then evaluate, on the held-out
tasks, how highly it ranks the correct program against a library of all unique
programs — a generalization test, compared against an untrained control and a
random baseline.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/michaelhodel/arc-dsl.git
git clone https://github.com/michaelhodel/re-arc.git
cd re-arc && unzip re_arc.zip -d re_arc_data && cd ..
```

## Run

```bash
python arc_data.py    # sanity-check the data assembly
python train.py       # train ARC-CLIP (saves arc_clip.pt, task_split.json, vocab.json)
python evaluate.py    # held-out retrieval evaluation
```

## Files

| File | Role |
|------|------|
| `arc_data.py` | assemble `(demos, program)` pairs from arc-dsl + re-arc |
| `encoding.py` | grids -> one-hot tensors; programs -> token ids |
| `model.py` | `phi_v` (CNN) and `phi_p` (Transformer) encoders |
| `train.py` | InfoNCE training with the held-out task split |
| `evaluate.py` | held-out retrieval evaluation with controls |

## Results

| Setting | median rank | top-1 | top-5 | top-10 | top-20 |
|---------|-------------|-------|-------|--------|--------|
| Held-out (test) | _fill in_ | | | | |
| Untrained control | | | | | |
| Train (reference) | | | | | |

_(Populate from `evaluate.py` output.)_
