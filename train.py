"""
train.py  (Phase 4)
-------------------
Train ARC-CLIP with a symmetric InfoNCE contrastive loss.

For each batch:
  1. encode all demo-sets with phi_v  -> v  (B, 128)
  2. encode all programs   with phi_p  -> p  (B, 128)
  3. build a (B, B) cosine-similarity matrix
  4. InfoNCE loss = "each demo-set should match its OWN program" (the diagonal)
  5. backpropagate and update the weights

We print the loss (should go DOWN) and the in-batch match accuracy
(should go UP, far above the random level of 1/B).
"""

import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from encode import demos_to_tensor, program_to_ids
from model import VisionEncoder, ProgramEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.07     # smaller = sharper similarities (CLIP's default)
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TARGET_DEMOS = 5       # we pad every example to exactly 5 demo pairs


# Pad an example's demos to exactly TARGET_DEMOS by cycling through them.
# All demos of an example show the SAME rule, so repeating them is harmless.
def pad_demos(d, target=TARGET_DEMOS):
    k = d.shape[0]
    if k >= target:
        return d[:target]
    idx = [i % k for i in range(target)]
    return d[idx]


class ARCPairs(Dataset):
    """Yields (demos_tensor, program_ids) for each synthetic example."""
    def __init__(self, path="synthetic_data.json"):
        self.data = json.load(open(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        demos = pad_demos(demos_to_tensor(ex))      # (5, 20, 30, 30)
        prog = program_to_ids(ex["program"])        # (4,)
        return demos, prog


def info_nce(v, p, prog_ids, temperature=TEMPERATURE):
    # v, p are length-1, so v @ p.T is the cosine-similarity matrix.
    logits = v @ p.t() / temperature                # (B, B)
    B = v.size(0)
    labels = torch.arange(B, device=v.device)       # correct match = diagonal

    # ---- collision fix --------------------------------------------------
    # Our small DSL means a batch often contains the SAME program twice.
    # Plain InfoNCE would punish those duplicates as "wrong" negatives, even
    # though they are actually correct. We find off-diagonal pairs that have
    # an identical program and remove them from the comparison (set their
    # similarity to -inf), so they count as neither positive nor negative.
    same = (prog_ids.unsqueeze(1) == prog_ids.unsqueeze(0)).all(dim=2)  # (B,B)
    duplicates = same & ~torch.eye(B, dtype=torch.bool, device=v.device)
    logits = logits.masked_fill(duplicates, float("-inf"))
    # ---------------------------------------------------------------------

    loss_demos_to_prog = F.cross_entropy(logits, labels)
    loss_prog_to_demos = F.cross_entropy(logits.t(), labels)
    loss = (loss_demos_to_prog + loss_prog_to_demos) / 2
    match_acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, match_acc


def main():
    torch.manual_seed(0)            # reproducible runs (less noise between runs)
    dataset = ARCPairs()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    phi_v = VisionEncoder().to(DEVICE)
    phi_p = ProgramEncoder().to(DEVICE)
    params = list(phi_v.parameters()) + list(phi_p.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    print(f"Training on {DEVICE} | {len(dataset)} examples | batch {BATCH_SIZE}")
    print(f"Random-guess match accuracy would be about {1/BATCH_SIZE:.3f}\n")

    for epoch in range(1, EPOCHS + 1):
        phi_v.train(); phi_p.train()
        total_loss = total_acc = n_batches = 0
        t0 = time.time()
        for demos, progs in loader:
            demos = demos.to(DEVICE)
            progs = progs.to(DEVICE)

            v = phi_v(demos)
            p = phi_p(progs)
            loss, acc = info_nce(v, p, progs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        print(f"epoch {epoch:2d} | loss {total_loss/n_batches:.4f} "
              f"| match acc {total_acc/n_batches:.3f} "
              f"| {time.time()-t0:.1f}s")

    torch.save({"phi_v": phi_v.state_dict(),
                "phi_p": phi_p.state_dict()}, "arc_clip.pt")
    print("\nSaved trained model -> arc_clip.pt")


if __name__ == "__main__":
    main()
