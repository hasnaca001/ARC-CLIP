# ARC-CLIP

A contrastive visual–symbolic model for the Abstraction and Reasoning Corpus
(ARC-AGI). The goal of **Objective 1** is to learn two encoders that map into a
shared embedding space:

- **phi_v** — encodes the visual transformation shown by a task's demonstration
  pairs.
- **phi_p** — encodes a DSL program (the symbolic rule).

Trained with a symmetric InfoNCE contrastive loss, matched (transformation,
program) pairs are pulled together and mismatched pairs pushed apart — the CLIP
idea, applied to ARC. A downstream **ARC-Solver** then uses the embeddings to
rank and verify candidate programs for unseen tasks.

## Project phases

| Phase | Module | Status |
|-------|--------|--------|
| 0 | `src/data.py` — load & visualize ARC tasks | done |
| 1 | synthetic `(demos, program)` generation | next |
| 2 | tensor encoding of grids and programs | |
| 3 | the phi_v and phi_p encoders | |
| 4 | InfoNCE training loop | |
| 5 | ARC-Solver: encode → rank → verify → apply | |

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/fchollet/ARC-AGI.git   # the dataset
python src/data.py                                   # self-test
```
