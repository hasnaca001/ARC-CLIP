"""
solve.py  (Objective 1, Part B) -- ARC-Solver
---------------------------------------------
Turn ARC-CLIP retrieval into an actual solver.

For each held-out task:
  1. RANK    candidate programs by cosine similarity to the demos (ARC-CLIP)
  2. VERIFY  each candidate by EXECUTING the real DSL program on the demos,
             keeping the first that reproduces every demonstration exactly
  3. APPLY   that program to the test input and check the predicted output

We report, within a fixed budget of program executions per task:
  - solve rate        : fraction of held-out tasks solved correctly
  - candidates checked : how many programs were executed before the answer
Compared against an untrained ranker and a random ranker -- the trained model
should solve MORE within the budget and with FEWER executions.

Run (after train.py):  python solve.py
"""

import sys
import json
import random
import statistics

sys.path.insert(0, "arc-dsl")
import solvers as SOLVERS            # the real arc-dsl solver functions

import torch

from arc_data import load_programs, load_task_examples
from encoding import program_to_ids, demos_to_tensor
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DEMOS = 5
BUDGET = 50                          # max programs executed per task


def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    return d[:target] if k >= target else d[[i % k for i in range(target)]]


def to_grid(g):
    return tuple(tuple(int(v) for v in row) for row in g)


def run_program(repr_task, grid):
    return getattr(SOLVERS, "solve_" + repr_task)(to_grid(grid))


def verifies(repr_task, demos):
    try:
        for d in demos:
            if run_program(repr_task, d["input"]) != to_grid(d["output"]):
                return False
        return True
    except Exception:
        return False


def build_library(programs, vocab):
    key_to_task = {}
    for t, ops in programs.items():
        key_to_task.setdefault(" ; ".join(ops), t)   # one representative task per program
    keys = list(key_to_task.keys())
    repr_tasks = [key_to_task[k] for k in keys]
    ids = torch.stack([program_to_ids(programs[key_to_task[k]], vocab) for k in keys])
    return repr_tasks, ids


def make_task(task_id):
    ex = load_task_examples(task_id, limit=300)
    k = random.randint(2, 5)
    picks = random.sample(ex, k + 1)
    demos = [{"input": e["input"], "output": e["output"]} for e in picks[:k]]
    return demos, picks[k]          # demos + one held-out test example


def solve_one(demos, test, phi_v, lib_emb, repr_tasks, order_fn):
    d = pad_demos(demos_to_tensor({"demos": demos})).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = phi_v(d)
        sims = (v @ lib_emb.t()).squeeze(0)
    order = order_fn(sims)
    for rank, idx in enumerate(order[:BUDGET], start=1):
        if verifies(repr_tasks[idx], demos):
            try:
                pred = run_program(repr_tasks[idx], test["input"])
            except Exception:
                continue
            return (pred == to_grid(test["output"])), rank
    return False, BUDGET


def run_setting(name, tasks, phi_v, lib_emb, repr_tasks, order_fn):
    solved, checks = 0, []
    for t in tasks:
        demos, test = make_task(t)
        ok, rank = solve_one(demos, test, phi_v, lib_emb, repr_tasks, order_fn)
        solved += int(ok)
        checks.append(rank)
    n = len(tasks)
    print(f"{name:20s} | solve rate {solved}/{n} = {solved/n:5.1%} "
          f"| avg programs run {statistics.mean(checks):4.1f} (budget {BUDGET})")


def main():
    random.seed(0)
    programs = load_programs()
    vocab = json.load(open("vocab.json"))
    test_tasks = json.load(open("task_split.json"))["test"]
    repr_tasks, lib_ids = build_library(programs, vocab)
    print(f"library: {len(repr_tasks)} programs | held-out tasks: {len(test_tasks)}\n")

    # trained ARC-CLIP
    ckpt = torch.load("arc_clip.pt", map_location=DEVICE, weights_only=False)
    phi_v = VisionEncoder().to(DEVICE); phi_v.load_state_dict(ckpt["phi_v"]); phi_v.eval()
    phi_p = ProgramEncoder(len(vocab)).to(DEVICE); phi_p.load_state_dict(ckpt["phi_p"]); phi_p.eval()
    with torch.no_grad():
        lib_emb = phi_p(lib_ids.to(DEVICE))

    # untrained control
    uv = VisionEncoder().to(DEVICE).eval()
    up = ProgramEncoder(len(vocab)).to(DEVICE).eval()
    with torch.no_grad():
        u_lib = up(lib_ids.to(DEVICE))

    cosine_order = lambda sims: torch.argsort(sims, descending=True).tolist()
    def random_order(sims):
        o = list(range(sims.numel())); random.shuffle(o); return o

    print("(each program is actually executed to verify -- this takes a few minutes)\n")
    run_setting("Trained ARC-CLIP", test_tasks, phi_v, lib_emb, repr_tasks, cosine_order)
    run_setting("Untrained control", test_tasks, uv, u_lib, repr_tasks, cosine_order)
    run_setting("Random order", test_tasks, phi_v, lib_emb, repr_tasks, random_order)


if __name__ == "__main__":
    main()
