"""
evaluate.py
-----------
The held-out evaluation -- ARC-CLIP's main result.

For each task, encode its demonstrations with phi_v and rank a library of all
unique real programs by cosine similarity; record where the true program lands.

Settings (same library throughout):
  Held-out (test) : tasks never seen in training   <-- the headline
  Untrained ctrl  : identical network, random weights
  Train (ref)     : training tasks (upper-bound reference)

Run (after train.py):  python evaluate.py
"""

import json
import random
import statistics

import torch

from arc_data import load_programs, sample_demo_set
from encoding import program_to_ids, demos_to_tensor
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DEMOS = 5
TRIALS = 3


def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    return d[:target] if k >= target else d[[i % k for i in range(target)]]


def build_library(programs, vocab):
    uniq = {}
    for ops in programs.values():
        uniq[" ; ".join(ops)] = ops
    keys = list(uniq.keys())
    ids = torch.stack([program_to_ids(uniq[k], vocab) for k in keys])
    return keys, ids


def evaluate(tasks, phi_v, lib_emb, programs, key_to_idx):
    ranks = []
    for t in tasks:
        true_idx = key_to_idx[" ; ".join(programs[t])]
        for _ in range(TRIALS):
            demos = pad_demos(demos_to_tensor(
                {"demos": sample_demo_set(t, random.randint(2, 5))})).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                v = phi_v(demos)
                sims = (v @ lib_emb.t()).squeeze(0)
            ranks.append(int((sims > sims[true_idx]).sum().item()) + 1)
    return ranks


def summarize(name, ranks, lib_size):
    n = len(ranks)
    topk = lambda k: sum(r <= k for r in ranks) / n
    print(f"{name:16s} | median rank {statistics.median(ranks):4.0f}/{lib_size} "
          f"| top-1 {topk(1):5.1%} | top-5 {topk(5):5.1%} "
          f"| top-10 {topk(10):5.1%} | top-20 {topk(20):5.1%}")


def main():
    random.seed(0)
    programs = load_programs()
    vocab = json.load(open("vocab.json"))
    split = json.load(open("task_split.json"))
    train_tasks, test_tasks = split["train"], split["test"]

    keys, lib_ids = build_library(programs, vocab)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    lib_size = len(keys)
    print(f"library: {lib_size} unique programs | held-out test tasks: {len(test_tasks)}\n")

    ckpt = torch.load("arc_clip.pt", map_location=DEVICE, weights_only=False)
    phi_v = VisionEncoder().to(DEVICE); phi_v.load_state_dict(ckpt["phi_v"]); phi_v.eval()
    phi_p = ProgramEncoder(len(vocab)).to(DEVICE); phi_p.load_state_dict(ckpt["phi_p"]); phi_p.eval()
    with torch.no_grad():
        lib_emb = phi_p(lib_ids.to(DEVICE))

    test_ranks = evaluate(test_tasks, phi_v, lib_emb, programs, key_to_idx)
    train_ref = evaluate(random.sample(train_tasks, min(80, len(train_tasks))),
                         phi_v, lib_emb, programs, key_to_idx)

    uv = VisionEncoder().to(DEVICE).eval()
    up = ProgramEncoder(len(vocab)).to(DEVICE).eval()
    with torch.no_grad():
        u_lib = up(lib_ids.to(DEVICE))
    ctrl_ranks = evaluate(test_tasks, uv, u_lib, programs, key_to_idx)

    print(f"{'setting':16s} | retrieval of the TRUE program (lower rank = better)")
    print("-" * 96)
    summarize("Held-out (test)", test_ranks, lib_size)
    summarize("Untrained ctrl", ctrl_ranks, lib_size)
    summarize("Train (ref)", train_ref, lib_size)
    print(f"\nRandom baseline: top-1 ~{1/lib_size:.1%}, top-10 ~{10/lib_size:.1%}, top-20 ~{20/lib_size:.1%}")


if __name__ == "__main__":
    main()
