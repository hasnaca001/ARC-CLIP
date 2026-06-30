"""
encoding.py
-----------
Turn ARC data into tensors for the network.

Grids    : one-hot over 10 colors on a fixed 30x30 canvas. A demo pair stacks
           its input and output -> 20 channels.
Programs : tokenize the arc-dsl operation strings into a learned vocabulary
           (~254 tokens) and pad to a fixed length.
"""

import re

import numpy as np
import torch

CANVAS = 30
NUM_COLORS = 10
MAX_PROG_TOKENS = 256


# --------------------------- grids -> tensors ------------------------------
def grid_to_tensor(grid, canvas=CANVAS):
    g = np.array(grid, dtype=np.int64)
    g = g[:canvas, :canvas]                      # crop anything larger than the canvas
    h, w = g.shape
    t = torch.zeros(NUM_COLORS, canvas, canvas, dtype=torch.float32)
    for color in range(NUM_COLORS):
        t[color, :h, :w] = torch.from_numpy((g == color)).float()
    return t


def pair_to_tensor(pair, canvas=CANVAS):
    return torch.cat([grid_to_tensor(pair["input"], canvas),
                      grid_to_tensor(pair["output"], canvas)], dim=0)   # (20, C, C)


def demos_to_tensor(example, canvas=CANVAS):
    return torch.stack([pair_to_tensor(p, canvas) for p in example["demos"]], dim=0)  # (k,20,C,C)


# ------------------------- programs -> token ids ---------------------------
def tokenize_program(ops):
    return re.findall(r"[A-Za-z_]\w*|\d+|[(),;]", " ; ".join(ops))


def build_vocab(programs):
    """Deterministic token -> id map from all programs."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for ops in programs.values():
        for tok in tokenize_program(ops):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def program_to_ids(ops, vocab, max_len=MAX_PROG_TOKENS):
    toks = tokenize_program(ops)
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)
