"""
realencode.py  (R2)
-------------------
Encode REAL ARC data for ARC-CLIP.

Grids   : reuse the 30x30 one-hot encoder from encode.py (real grids are <=30x30).
Programs: tokenize arc-dsl operation sequences into a learned vocabulary.
          The vocabulary has 254 tokens -- the 160 DSL primitives plus the
          variables (I, x1, x2, ...), constants (F, T, ONE, ...) and punctuation.
"""

import re
import statistics

import torch

from realdata import load_programs, list_task_ids, sample_demo_set
from encode import demos_to_tensor          # reuse grid encoding (30x30 one-hot)

MAX_PROG_TOKENS = 256        # covers ~90% of programs; the longest ones are truncated


def tokenize_program(ops):
    """Flatten a list of operation strings into a list of tokens.

    e.g. ['objects(I, F, F, T)', 'argmax(x1, numcolors)']
      -> ['objects','(','I',',','F',',','F',',','T',')',';','argmax','(', ...]
    """
    text = " ; ".join(ops)
    return re.findall(r"[A-Za-z_]\w*|\d+|[(),;]", text)


def build_vocab(programs):
    """Deterministic token -> id map built from all 400 programs."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for ops in programs.values():
        for tok in tokenize_program(ops):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def program_to_ids(ops, vocab, max_len=MAX_PROG_TOKENS):
    toks = tokenize_program(ops)
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))   # pad to fixed length
    return torch.tensor(ids, dtype=torch.long)


if __name__ == "__main__":
    programs = load_programs()
    vocab = build_vocab(programs)
    print("program vocabulary size:", len(vocab))

    lens = [len(tokenize_program(o)) for o in programs.values()]
    print("program token length: median %d, max %d" % (int(statistics.median(lens)), max(lens)))

    # sanity check: encode one task's demos and program into tensors
    ids = list_task_ids()
    t = ids[0]
    demos = sample_demo_set(t, 3)
    demos_tensor = demos_to_tensor({"demos": demos})    # (3, 20, 30, 30)
    prog_ids = program_to_ids(programs[t], vocab)        # (256,)
    print(f"task {t}: demos tensor {tuple(demos_tensor.shape)}, "
          f"program ids {tuple(prog_ids.shape)}")
