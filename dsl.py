"""
src/dsl.py  (Phase 1)
---------------------
A tiny DSL and a generator of synthetic (demonstrations, program) pairs.

The plan:
  1. PRIMITIVES: simple operations, each turns one grid into another grid.
  2. A "program" is just a short list of primitive names, e.g. ["flip_h", "rot90"].
  3. apply_program runs a program on a grid.
  4. make_example: pick a random program, make a few random input grids,
     run the program on each to get the outputs. That gives us one
     (demonstrations, program) training example -- the program is the "label".
  5. make_dataset: do that thousands of times and save to a JSON file.

This is the training data ARC-CLIP will learn from in later phases.
"""

import json
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# 1. THE PRIMITIVES
# Each function takes a 2D numpy array (a grid) and returns a transformed grid.
# We keep them parameter-free so a program is just a simple list of names.
# ---------------------------------------------------------------------------
def identity(g):  return g
def flip_h(g):    return g[:, ::-1]      # mirror left<->right
def flip_v(g):    return g[::-1, :]      # mirror top<->bottom
def rot90(g):     return np.rot90(g, 1)  # rotate 90 degrees
def rot180(g):    return np.rot90(g, 2)
def rot270(g):    return np.rot90(g, 3)
def transpose(g): return g.T             # flip across the main diagonal

PRIMITIVES = {
    "identity":  identity,
    "flip_h":    flip_h,
    "flip_v":    flip_v,
    "rot90":     rot90,
    "rot180":    rot180,
    "rot270":    rot270,
    "transpose": transpose,
}

# ---------------------------------------------------------------------------
# 2 & 3. APPLY A PROGRAM (a list of primitive names) TO A GRID
# ---------------------------------------------------------------------------
def apply_program(program, grid):
    g = np.array(grid, dtype=np.int64)
    for name in program:
        g = PRIMITIVES[name](g)
    return g.tolist()   # convert back to plain lists so it can be saved as JSON


# ---------------------------------------------------------------------------
# 4a. MAKE A RANDOM INPUT GRID
# A sparse grid: mostly black (0) with a few randomly colored cells.
# ---------------------------------------------------------------------------
def random_grid(min_size=3, max_size=8):
    h = random.randint(min_size, max_size)
    w = random.randint(min_size, max_size)
    g = np.zeros((h, w), dtype=np.int64)
    n_cells = random.randint(2, max(3, (h * w) // 3))
    colors = random.sample(range(1, 10), k=random.randint(2, 4))
    for _ in range(n_cells):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        g[r, c] = random.choice(colors)
    return g.tolist()


# ---------------------------------------------------------------------------
# 4b. MAKE ONE TRAINING EXAMPLE
# A random program + k demonstration pairs created by that program.
# ---------------------------------------------------------------------------
def make_example(min_len=1, max_len=2):
    names = [n for n in PRIMITIVES if n != "identity"]
    program = [random.choice(names) for _ in range(random.randint(min_len, max_len))]
    k = random.randint(2, 5)                       # ARC tasks have 2..5 demos
    demos = []
    for _ in range(k):
        inp = random_grid()
        out = apply_program(program, inp)
        demos.append({"input": inp, "output": out})
    return {"program": program, "demos": demos}


# ---------------------------------------------------------------------------
# 5. MAKE A WHOLE DATASET AND SAVE IT
# ---------------------------------------------------------------------------
def make_dataset(n, path="synthetic_data.json", seed=0):
    random.seed(seed)
    np.random.seed(seed)
    data = [make_example() for _ in range(n)]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Generated {n} examples -> {path}")
    return data


# ---------------------------------------------------------------------------
# A helper to LOOK at one generated example (same colors as real ARC).
# ---------------------------------------------------------------------------
ARC_COLORS = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
              "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
CMAP = ListedColormap(ARC_COLORS)

def plot_example(example, save_path=None):
    demos = example["demos"]
    fig, axes = plt.subplots(2, len(demos), figsize=(2.2 * len(demos), 4.5))
    if len(demos) == 1:
        axes = axes.reshape(2, 1)
    for col, d in enumerate(demos):
        for row, which in enumerate(["input", "output"]):
            g = np.array(d[which])
            ax = axes[row][col]
            ax.imshow(g, cmap=CMAP, vmin=0, vmax=9)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel(which)
            if row == 0: ax.set_title(f"demo {col + 1}")
    plt.suptitle("program: " + "  ->  ".join(example["program"]))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
        print("saved figure to", save_path)
    else:
        plt.show()


if __name__ == "__main__":
    data = make_dataset(2000)
    ex = data[0]
    print("First example's program:", ex["program"])
    print("It has", len(ex["demos"]), "demonstration pairs.")
    plot_example(ex, save_path="synthetic_example.png")
