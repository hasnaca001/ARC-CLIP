"""
solve.py  (Phase 5) -- ARC-Solver
---------------------------------
Use the trained ARC-CLIP model to solve unseen tasks.

Pipeline for one task (matches section 6.2.5 of the report):
  1. ENCODE  : v = phi_v(demos)
  2. RANK    : score every candidate program by cosine similarity to v
  3. VERIFY  : run programs in ranked order; keep the first that reproduces
               ALL demonstration outputs exactly
  4. APPLY   : run that program on the test input -> the prediction

Two things are measured:
  - solve rate     : did we find a program that also gets the TEST right
  - search efficiency : how far down the ranked list the answer was
                        (low = the learned embedding is a good heuristic)
"""

import itertools
import statistics
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from dsl import PRIMITIVES, apply_program, make_example, random_grid, CMAP
from encode import demos_to_tensor, program_to_ids
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DEMOS = 5


def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    if k >= target:
        return d[:target]
    return d[[i % k for i in range(target)]]


# 1. Build the candidate library: every program of length 1 and 2.
def build_candidates(max_len=2):
    names = list(PRIMITIVES.keys())
    progs = []
    for L in range(1, max_len + 1):
        for combo in itertools.product(names, repeat=L):
            progs.append(list(combo))
    return progs


# 2. Load the trained encoders.
def load_model():
    ckpt = torch.load("arc_clip.pt", map_location=DEVICE, weights_only=False)
    phi_v = VisionEncoder().to(DEVICE); phi_v.load_state_dict(ckpt["phi_v"]); phi_v.eval()
    phi_p = ProgramEncoder().to(DEVICE); phi_p.load_state_dict(ckpt["phi_p"]); phi_p.eval()
    return phi_v, phi_p


# A program "verifies" if it reproduces every demonstration output exactly.
def verifies(program, demos):
    return all(apply_program(program, d["input"]) == d["output"] for d in demos)


# 3 + 4. Solve one task: rank candidates, verify in order, return the winner.
def solve(task, phi_v, candidates, cand_emb):
    demos = task["demos"]
    d = pad_demos(demos_to_tensor(task)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = phi_v(d)                          # (1, 128)
        sims = (v @ cand_emb.t()).squeeze(0)  # similarity to every candidate
    order = torch.argsort(sims, descending=True).tolist()
    for rank, idx in enumerate(order, start=1):
        prog = candidates[idx]
        if verifies(prog, demos):
            return prog, rank                 # rank = how many we had to check
    return None, None


# Build a fresh held-out task the model has never seen, WITH a test pair.
def make_test_task():
    task = make_example()
    test_in = random_grid()
    task["test"] = {"input": test_in, "output": apply_program(task["program"], test_in)}
    return task


def plot_solution(task, found_prog, save_path):
    demos = task["demos"]
    cols = len(demos) + 1
    fig, axes = plt.subplots(2, cols, figsize=(2.2 * cols, 4.5))
    for c, d in enumerate(demos):
        for r, which in enumerate(["input", "output"]):
            axes[r][c].imshow(np.array(d[which]), cmap=CMAP, vmin=0, vmax=9)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if c == 0: axes[r][c].set_ylabel(which)
            if r == 0: axes[r][c].set_title(f"demo {c+1}")
    pred = apply_program(found_prog, task["test"]["input"])
    axes[0][-1].imshow(np.array(task["test"]["input"]), cmap=CMAP, vmin=0, vmax=9)
    axes[0][-1].set_title("TEST in"); axes[0][-1].set_xticks([]); axes[0][-1].set_yticks([])
    axes[1][-1].imshow(np.array(pred), cmap=CMAP, vmin=0, vmax=9)
    axes[1][-1].set_title("PREDICTED"); axes[1][-1].set_xticks([]); axes[1][-1].set_yticks([])
    plt.suptitle(f"true rule: {' -> '.join(task['program'])}   |   "
                 f"solver found: {' -> '.join(found_prog)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    print("saved figure to", save_path)


if __name__ == "__main__":
    random.seed(123)
    phi_v, phi_p = load_model()
    candidates = build_candidates()

    # Pre-embed every candidate program once.
    with torch.no_grad():
        cand_ids = torch.stack([program_to_ids(p) for p in candidates]).to(DEVICE)
        cand_emb = phi_p(cand_ids)            # (num_candidates, 128)
    print("Candidate program library size:", len(candidates))

    # Evaluate on fresh held-out tasks.
    N = 300
    solved, ranks = 0, []
    for _ in range(N):
        task = make_test_task()
        prog, rank = solve(task, phi_v, candidates, cand_emb)
        if prog is not None:
            ranks.append(rank)
            pred = apply_program(prog, task["test"]["input"])
            if pred == task["test"]["output"]:
                solved += 1

    print(f"\nSolve rate: {solved}/{N} = {solved/N:.1%}")
    print(f"Median candidates checked before the answer: {statistics.median(ranks)}")
    print(f"Mean candidates checked: {statistics.mean(ranks):.1f}  "
          f"(random-order search would need ~{len(candidates)/2:.0f})")

    # Save a picture of one solved task.
    task = make_test_task()
    prog, _ = solve(task, phi_v, candidates, cand_emb)
    plot_solution(task, prog, "solved_example.png")
