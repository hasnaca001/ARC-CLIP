"""
model.py  (Phase 3)
-------------------
The two encoders that make up ARC-CLIP.

  phi_v  (VisionEncoder)  : demos tensor (k, 20, 30, 30)  -> 128-dim vector
  phi_p  (ProgramEncoder) : program ids (max_len,)        -> 128-dim vector

Both outputs are L2-normalized (length 1) so we can compare them with cosine
similarity in the training phase. Nothing is trained yet -- this file just
defines the network shapes and proves they run and produce the right sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import VOCAB_SIZE

EMBED_DIM = 128   # the size of the shared embedding space


# ---------------------------------------------------------------------------
# phi_v : the VISION encoder (a small CNN over the demonstration pairs)
# ---------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, d=EMBED_DIM):
        super().__init__()
        # A CNN reads one demo pair (20 channels) and produces a 64-num feature.
        self.cnn = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 30 -> 15
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 15 -> 8
            nn.AdaptiveAvgPool2d(1),                                           # -> (64,1,1)
        )
        self.proj = nn.Linear(64, d)

    def forward(self, demos):
        # demos shape: (B, k, 20, 30, 30)   B = batch size, k = number of demos
        B, k, C, H, W = demos.shape
        x = demos.view(B * k, C, H, W)          # treat every demo as its own image
        feat = self.cnn(x).view(B * k, 64)      # (B*k, 64)
        feat = feat.view(B, k, 64).mean(dim=1)  # AVERAGE over the k demos -> (B, 64)
        v = self.proj(feat)                     # (B, 128)
        return F.normalize(v, dim=-1)           # make each vector length 1


# ---------------------------------------------------------------------------
# phi_p : the PROGRAM encoder (an embedding + a GRU that reads the sequence)
# ---------------------------------------------------------------------------
class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d=EMBED_DIM, emb=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)      # each token id -> 32-num vector
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d)

    def forward(self, prog_ids):
        # prog_ids shape: (B, max_len)
        e = self.embed(prog_ids)        # (B, max_len, 32)
        _, h = self.gru(e)              # h is the final summary: (1, B, 64)
        p = self.proj(h.squeeze(0))     # (B, 128)
        return F.normalize(p, dim=-1)


# ---------------------------------------------------------------------------
# SELF-TEST: run both encoders on one real example and check the output sizes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from encode import demos_to_tensor, program_to_ids

    data = json.load(open("synthetic_data.json"))
    ex = data[0]

    # add a batch dimension of 1 to each input
    demos = demos_to_tensor(ex).unsqueeze(0)            # (1, k, 20, 30, 30)
    progs = program_to_ids(ex["program"]).unsqueeze(0)  # (1, 4)

    phi_v = VisionEncoder()
    phi_p = ProgramEncoder()

    v = phi_v(demos)
    p = phi_p(progs)

    print("vision embedding shape :", tuple(v.shape))    # (1, 128)
    print("program embedding shape:", tuple(p.shape))    # (1, 128)

    cos = (v * p).sum(dim=-1)   # cosine similarity (both are length 1)
    print("cosine similarity BEFORE training:", round(float(cos), 4),
          "  (near 0 = random, as expected for an untrained model)")

    n_v = sum(x.numel() for x in phi_v.parameters())
    n_p = sum(x.numel() for x in phi_p.parameters())
    print(f"phi_v parameters: {n_v:,}   phi_p parameters: {n_p:,}")
