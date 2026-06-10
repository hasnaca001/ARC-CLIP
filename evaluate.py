"""
evaluate.py
-----------
Does ARC-CLIP actually work? This compares the trained model against two
controls on the SAME held-out tasks, using the SAME verify-by-execution solver.

Rankers compared:
  1. Trained ARC-CLIP   -- your learned model
  2. Untrained (control) -- identical network, random weights (no learning)
  3. Random order        -- no model at all

Metric: for each task, how far down the ranked candidate list is the FIRST
program that verifies (reproduces all demos)? This is the number of programs
the solver must execute -- lower is better.

If the trained model needs far fewer checks and has much higher top-1 than the
controls, the learned visual->program heuristic is real. If all three are
similar, the model learned nothing useful.
"""

import itertools
import statistics
import random

import torch
import matplotlib.pyplot as plt

from dsl import PRIMITIVES, apply_program, make_example, random_grid
from encode import demos_to_tensor, program_to_ids
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DEMOS = 5


def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    return d[:target] if k >= target else d[[i % k for i in range(target)]]


def build_candidates(max_len=2):
    names = list(PRIMITIVES.keys())
    return [list(c) for L in range(1, max_len + 1)
            for c in itertools.product(names, repeat=L)]


def verifies(program, demos):
    return all(apply_program(program, d["input"]) == d["output"] for d in demos)


def make_test_task():
    task = make_example()
    test_in = random_grid()
    task["test"] = {"input": test_in,
                    "output": apply_program(task["program"], test_in)}
    return task


def rank_of_first_verifier(order, candidates, demos):
    for rank, idx in enumerate(order, start=1):
        if verifies(candidates[idx], demos):
            return rank
    return len(candidates)   # should never happen (true program is in the set)


def embed_candidates(phi_p, candidates):
    with torch.no_grad():
        ids = torch.stack([program_to_ids(p) for p in candidates]).to(DEVICE)
        return phi_p(ids)


def cosine_order(task, phi_v, cand_emb):
    d = pad_demos(demos_to_tensor(task)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = phi_v(d)
        sims = (v @ cand_emb.t()).squeeze(0)
    return torch.argsort(sims, descending=True).tolist()


def collect_ranks(order_fn, tasks, candidates):
    return [rank_of_first_verifier(order_fn(t), candidates, t["demos"]) for t in tasks]


def summarize(name, ranks):
    n = len(ranks)
    top1 = sum(r == 1 for r in ranks) / n
    top3 = sum(r <= 3 for r in ranks) / n
    print(f"{name:22s} | mean checked {statistics.mean(ranks):5.1f} "
          f"| median {statistics.median(ranks):4.0f} "
          f"| top-1 {top1:6.1%} | top-3 {top3:6.1%}")


if __name__ == "__main__":
    random.seed(999)
    candidates = build_candidates()
    tasks = [make_test_task() for _ in range(300)]
    print(f"Candidates: {len(candidates)} | held-out tasks: {len(tasks)} "
          f"| blind search would average ~{len(candidates)/2:.0f} checks\n")

    # ---- Trained ARC-CLIP ----
    ckpt = torch.load("arc_clip.pt", map_location=DEVICE, weights_only=False)
    tv = VisionEncoder().to(DEVICE); tv.load_state_dict(ckpt["phi_v"]); tv.eval()
    tp = ProgramEncoder().to(DEVICE); tp.load_state_dict(ckpt["phi_p"]); tp.eval()
    trained_ranks = collect_ranks(lambda t: cosine_order(t, tv, embed_candidates(tp, candidates)),
                                  tasks, candidates)

    # ---- Untrained control (random weights, same architecture) ----
    uv = VisionEncoder().to(DEVICE); uv.eval()
    up = ProgramEncoder().to(DEVICE); up.eval()
    u_emb = embed_candidates(up, candidates)
    untrained_ranks = collect_ranks(lambda t: cosine_order(t, uv, u_emb), tasks, candidates)

    # ---- Random order ----
    def random_order(_):
        o = list(range(len(candidates))); random.shuffle(o); return o
    random_ranks = collect_ranks(random_order, tasks, candidates)

    print(f"{'ranker':22s} | {'search cost (lower=better)':35s} | retrieval")
    print("-" * 92)
    summarize("Trained ARC-CLIP", trained_ranks)
    summarize("Untrained (control)", untrained_ranks)
    summarize("Random order", random_ranks)

    # ---- Picture: how the search cost is distributed ----
    plt.figure(figsize=(8, 4))
    bins = range(1, len(candidates) + 2)
    plt.hist(trained_ranks, bins=bins, alpha=0.7, label="Trained ARC-CLIP")
    plt.hist(untrained_ranks, bins=bins, alpha=0.5, label="Untrained control")
    plt.xlabel("candidates checked before the answer (lower = better)")
    plt.ylabel("number of tasks")
    plt.title("Search cost: trained model vs untrained control")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=110)
    print("\nsaved figure to evaluation.png")
