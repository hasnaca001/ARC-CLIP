"""
train.py
--------
Train ARC-CLIP with a symmetric InfoNCE contrastive loss.

The 400 tasks are split into TRAIN (~320) and held-out TEST (~80). Training uses
only train tasks; evaluate.py later measures retrieval on the held-out tasks the
model never saw. Saves: arc_clip.pt, task_split.json, vocab.json.

Run:  python train.py
"""

import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arc_data import load_programs, list_task_ids, load_task_examples
from encoding import build_vocab, program_to_ids, demos_to_tensor
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.07
EPOCHS = 30
BATCH_SIZE = 64
LR = 3e-4
TARGET_DEMOS = 5
SAMPLES_PER_EPOCH = 6400
TEST_FRACTION = 0.2
MAX_PER_TASK = 200          # cache this many of the 1000 examples per task


def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    return d[:target] if k >= target else d[[i % k for i in range(target)]]


def split_tasks(task_ids, test_fraction=TEST_FRACTION, seed=0):
    ids = list(task_ids)
    random.Random(seed).shuffle(ids)
    n_test = int(len(ids) * test_fraction)
    return ids[n_test:], ids[:n_test]


class ARCPairs(Dataset):
    def __init__(self, tasks, programs, vocab, n):
        self.tasks, self.n = tasks, n
        self.cache, self.prog_ids = {}, {}
        for t in tasks:
            ex = load_task_examples(t, limit=MAX_PER_TASK)
            self.cache[t] = [(np.array(e["input"], dtype=np.int8),
                              np.array(e["output"], dtype=np.int8)) for e in ex]
            self.prog_ids[t] = program_to_ids(programs[t], vocab)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t = random.choice(self.tasks)
        pairs = random.sample(self.cache[t], random.randint(2, 5))
        demos = pad_demos(demos_to_tensor(
            {"demos": [{"input": a, "output": b} for a, b in pairs]}))
        return demos, self.prog_ids[t]


def info_nce(v, p, prog_ids, temperature=TEMPERATURE):
    logits = v @ p.t() / temperature
    B = v.size(0)
    labels = torch.arange(B, device=v.device)
    same = (prog_ids.unsqueeze(1) == prog_ids.unsqueeze(0)).all(dim=2)
    dup = same & ~torch.eye(B, dtype=torch.bool, device=v.device)   # mask duplicate programs
    logits = logits.masked_fill(dup, float("-inf"))
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def main():
    torch.manual_seed(0)
    random.seed(0)

    programs = load_programs()
    vocab = build_vocab(programs)
    all_ids = [t for t in list_task_ids() if t in programs]
    train_tasks, test_tasks = split_tasks(all_ids)
    print(f"tasks: {len(all_ids)} -> {len(train_tasks)} train, {len(test_tasks)} held-out test")

    json.dump({"train": train_tasks, "test": test_tasks}, open("task_split.json", "w"))
    json.dump(vocab, open("vocab.json", "w"))

    print("caching examples in memory (one-time)...")
    ds = ARCPairs(train_tasks, programs, vocab, SAMPLES_PER_EPOCH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    phi_v = VisionEncoder().to(DEVICE)
    phi_p = ProgramEncoder(len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(list(phi_v.parameters()) + list(phi_p.parameters()), lr=LR)

    print(f"training on {DEVICE}\n")
    for epoch in range(1, EPOCHS + 1):
        phi_v.train(); phi_p.train()
        tl = ta = n = 0
        t0 = time.time()
        for demos, progs in dl:
            demos, progs = demos.to(DEVICE), progs.to(DEVICE)
            v, p = phi_v(demos), phi_p(progs)
            loss, acc = info_nce(v, p, progs)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); ta += acc.item(); n += 1
        print(f"epoch {epoch:2d} | loss {tl/n:.4f} | in-batch acc {ta/n:.3f} | {time.time()-t0:.1f}s")

    torch.save({"phi_v": phi_v.state_dict(), "phi_p": phi_p.state_dict()}, "arc_clip.pt")
    print("\nsaved -> arc_clip.pt, task_split.json, vocab.json")


if __name__ == "__main__":
    main()
