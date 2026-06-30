"""
model.py
--------
The two ARC-CLIP encoders, both projecting to a shared 128-dim space and
L2-normalized for cosine-similarity / InfoNCE alignment.

  VisionEncoder  (phi_v) : CNN over demonstration pairs (B, k, 20, 30, 30) -> (B, 128)
  ProgramEncoder (phi_p) : Transformer over program token ids (B, L)       -> (B, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 128


class VisionEncoder(nn.Module):
    def __init__(self, d=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 30 -> 15
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 15 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, d)

    def forward(self, demos):                       # (B, k, 20, 30, 30)
        B, k, C, H, W = demos.shape
        x = demos.view(B * k, C, H, W)
        feat = self.cnn(x).view(B * k, 64)
        feat = feat.view(B, k, 64).mean(dim=1)      # permutation-invariant over demos
        return F.normalize(self.proj(feat), dim=-1)


class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size, d=EMBED_DIM, dim=128, nhead=4, layers=2, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                           dim_feedforward=256, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(dim, d)

    def forward(self, prog_ids):                    # (B, L)
        B, L = prog_ids.shape
        pad = (prog_ids == 0)
        pos = torch.arange(L, device=prog_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(prog_ids) + self.pos(pos)
        h = self.encoder(x, src_key_padding_mask=pad)
        m = (~pad).unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)   # mean over real tokens
        return F.normalize(self.proj(pooled), dim=-1)
