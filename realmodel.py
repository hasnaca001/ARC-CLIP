"""
realmodel.py  (R3)
------------------
The two encoders for the real ARC-CLIP.

  phi_v : a CNN over demonstration pairs (the proven design)   -> 128-dim vector
  phi_p : a small Transformer over the 256-token arc-dsl program -> 128-dim vector

Both are L2-normalized so we can align them with cosine similarity / InfoNCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 128


# ---------------------------------------------------------------------------
# phi_v : vision encoder over demonstration pairs (same proven CNN)
# ---------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, d=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 30 -> 15
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 15 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, d)

    def forward(self, demos):                      # (B, k, 20, 30, 30)
        B, k, C, H, W = demos.shape
        x = demos.view(B * k, C, H, W)
        feat = self.cnn(x).view(B * k, 64)
        feat = feat.view(B, k, 64).mean(dim=1)     # average over demos
        return F.normalize(self.proj(feat), dim=-1)


# ---------------------------------------------------------------------------
# phi_p : program encoder -- a small Transformer over the token sequence
# ---------------------------------------------------------------------------
class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size, d=EMBED_DIM, dim=128, nhead=4, layers=2, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=0)   # token -> vector
        self.pos = nn.Embedding(max_len, dim)                     # position -> vector
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(dim, d)

    def forward(self, prog_ids):                   # (B, L)
        B, L = prog_ids.shape
        pad = (prog_ids == 0)                       # True where padding
        pos = torch.arange(L, device=prog_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(prog_ids) + self.pos(pos)
        h = self.encoder(x, src_key_padding_mask=pad)     # (B, L, dim)
        # mean-pool over the real (non-pad) tokens
        m = (~pad).unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


if __name__ == "__main__":
    from realdata import load_programs, list_task_ids, sample_demo_set
    from realencode import build_vocab, program_to_ids, demos_to_tensor

    programs = load_programs()
    vocab = build_vocab(programs)
    t = list_task_ids()[0]

    demos = demos_to_tensor({"demos": sample_demo_set(t, 3)}).unsqueeze(0)   # (1,3,20,30,30)
    prog = program_to_ids(programs[t], vocab).unsqueeze(0)                    # (1,256)

    phi_v = VisionEncoder()
    phi_p = ProgramEncoder(len(vocab))
    v, p = phi_v(demos), phi_p(prog)

    print("vision embedding :", tuple(v.shape))
    print("program embedding:", tuple(p.shape))
    print("cosine (untrained):", round(float((v * p).sum(-1)), 4))
    print("phi_p parameters:", f"{sum(x.numel() for x in phi_p.parameters()):,}")
