"""
src/data.py
-----------
Foundation module for the ARC-CLIP project: everything related to loading
and looking at ARC tasks. You will keep reusing these functions in every
later phase, so it is worth keeping them clean and in one place.

A quick recap of the data:
  - An ARC task is a .json file with two keys: "train" and "test".
  - "train" is the list of DEMONSTRATION pairs the solver may learn from.
  - "test"  is the pair(s) whose output must be predicted.
  - Each pair is {"input": grid, "output": grid}.
  - A grid is a list of rows; each row is a list of ints 0..9 (colors).
"""

import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# The 10 official ARC colors, indexed 0..9 (0 = black background).
ARC_COLORS = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
              "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
CMAP = ListedColormap(ARC_COLORS)


def load_task(path):
    """Read one ARC task .json file and return it as a Python dict."""
    with open(path) as f:
        return json.load(f)


def list_tasks(folder):
    """Return a sorted list of every .json task path inside a folder."""
    return sorted(glob.glob(os.path.join(folder, "*.json")))


def grid_to_array(grid):
    """Convert a grid (list of rows) into a 2D numpy array of integers."""
    return np.array(grid, dtype=np.int64)


def plot_task(task, title="ARC task", save_path=None):
    """Draw all demonstration pairs of a task as a grid of pictures.

    If save_path is given, the figure is written to that file; otherwise
    it is shown on screen (which is what you want inside a notebook).
    """
    pairs = task["train"]
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.2 * len(pairs), 4.5))

    # Guard against the one-pair case so indexing always works.
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, pair in enumerate(pairs):
        for row, which in enumerate(["input", "output"]):
            g = grid_to_array(pair[which])
            ax = axes[row][col]
            ax.imshow(g, cmap=CMAP, vmin=0, vmax=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(which)
            if row == 0:
                ax.set_title(f"demo {col + 1}")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
        print("saved figure to", save_path)
    else:
        plt.show()


# This block only runs if you execute "python src/data.py" directly.
# It is a tiny self-test so you can confirm the module works.
if __name__ == "__main__":
    DATA_DIR = "ARC-AGI/data/training"
    tasks = list_tasks(DATA_DIR)
    print("Found", len(tasks), "training tasks.")
    task = load_task(tasks[0])
    print("First task has", len(task["train"]), "demo pairs.")
    plot_task(task, title="Self-test task", save_path="selftest.png")
