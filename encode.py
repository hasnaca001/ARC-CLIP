"""
encode.py  (Phase 2)
--------------------
Translate ARC data into tensors that a neural network can read.

Two jobs:
  A) grids   -> one-hot tensors on a fixed 30x30 canvas
  B) programs -> lists of integer token ids (padded to a fixed length)

Nothing here learns anything yet; it is pure data conversion. We also include
a decode function so we can PROVE the encoding loses no information.
"""

import numpy as np
import torch

from dsl import PRIMITIVES   # reuse the primitive names defined in Phase 1

CANVAS = 30        # ARC grids are at most 30x30
NUM_COLORS = 10    # colors are 0..9
MAX_PROG_LEN = 4   # longest program we will represent


# ---------------------------------------------------------------------------
# JOB A: GRID  ->  TENSOR  of shape (10, CANVAS, CANVAS)
# Each color channel is 1 where the grid has that color, else 0.
# Cells outside the real grid stay all-zero (that marks them as "padding").
# ---------------------------------------------------------------------------
def grid_to_tensor(grid, canvas=CANVAS):
    g = np.array(grid, dtype=np.int64)
    h, w = g.shape
    t = torch.zeros(NUM_COLORS, canvas, canvas, dtype=torch.float32)
    for color in range(NUM_COLORS):
        mask = (g == color)                       # True where this color appears
        t[color, :h, :w] = torch.from_numpy(mask).float()
    return t


# A demo pair = its input tensor stacked on top of its output tensor.
# Shape: (20, CANVAS, CANVAS)  -> first 10 channels = input, next 10 = output.
def pair_to_tensor(pair, canvas=CANVAS):
    inp = grid_to_tensor(pair["input"], canvas)
    out = grid_to_tensor(pair["output"], canvas)
    return torch.cat([inp, out], dim=0)


# A full example's demos -> one tensor of shape (k, 20, CANVAS, CANVAS),
# where k is how many demonstration pairs that example has (2..5).
def demos_to_tensor(example, canvas=CANVAS):
    pairs = [pair_to_tensor(p, canvas) for p in example["demos"]]
    return torch.stack(pairs, dim=0)


# ---------------------------------------------------------------------------
# JOB B: PROGRAM  ->  TOKEN IDS
# Build a vocabulary: id 0 is reserved for padding, then one id per primitive.
# ---------------------------------------------------------------------------
PROGRAM_VOCAB = ["<pad>"] + list(PRIMITIVES.keys())   # e.g. ['<pad>','identity','flip_h',...]
TOK2ID = {name: i for i, name in enumerate(PROGRAM_VOCAB)}
ID2TOK = {i: name for name, i in TOK2ID.items()}
VOCAB_SIZE = len(PROGRAM_VOCAB)

def program_to_ids(program, max_len=MAX_PROG_LEN):
    ids = [TOK2ID[name] for name in program][:max_len]
    ids += [TOK2ID["<pad>"]] * (max_len - len(ids))   # pad to fixed length
    return torch.tensor(ids, dtype=torch.long)

def ids_to_program(ids):
    return [ID2TOK[int(i)] for i in ids if int(i) != TOK2ID["<pad>"]]


# ---------------------------------------------------------------------------
# DECODE (for verification only): tensor -> grid, to confirm we lost nothing.
# ---------------------------------------------------------------------------
def tensor_to_grid(t):
    occupied = t.sum(dim=0) > 0          # real cells; padding is all-zero
    colors = t.argmax(dim=0)             # which color each cell is
    rows = torch.any(occupied, dim=1)
    cols = torch.any(occupied, dim=0)
    h = int(rows.nonzero().max()) + 1
    w = int(cols.nonzero().max()) + 1
    return colors[:h, :w].tolist()


# ---------------------------------------------------------------------------
# SELF-TEST: load the synthetic data, encode one example, check the shapes,
# and prove the grid encoding round-trips exactly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    data = json.load(open("synthetic_data.json"))
    ex = data[0]
    print("Program (words):", ex["program"])

    prog_ids = program_to_ids(ex["program"])
    print("Program (ids)  :", prog_ids.tolist())
    print("Back to words  :", ids_to_program(prog_ids))

    demos = demos_to_tensor(ex)
    print("Demos tensor shape (k, 20, 30, 30):", tuple(demos.shape))

    one_grid = ex["demos"][0]["input"]
    recovered = tensor_to_grid(grid_to_tensor(one_grid))
    print("Encoding is lossless:", recovered == one_grid)
    print("Vocabulary size:", VOCAB_SIZE, "->", PROGRAM_VOCAB)
