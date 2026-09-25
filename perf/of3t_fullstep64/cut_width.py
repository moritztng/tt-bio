#!/usr/bin/env python3
"""of3t-fullstep64 padding discriminator: the pinned batch cut to a narrower token width.

    cut_width.py batch_step003.pt W out.pt

Every token axis (length 384) is cut to [:W]. The pinned batch has its 56 real tokens first and
every padded token all-zero except `msa` (gap class 31), no padded atoms (422 real) and no
template, so the cut drops padding only; `ground_truth` is kept whole. The rollout draws are
per atom, so the 384-wide run's draws.pt replays unchanged at any W.
"""
import sys

import torch

src, W, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
b = torch.load(src, weights_only=False)
assert bool((b["token_mask"][0, :56] == 1).all()) and float(b["token_mask"][0, 56:].sum()) == 0


def cut(x):
    if isinstance(x, dict):
        return {k: cut(v) for k, v in x.items()}
    if torch.is_tensor(x):
        return x[tuple(slice(0, W) if s == 384 else slice(None) for s in x.shape)].clone()
    return x


t = cut(b)
t["ground_truth"] = b["ground_truth"]
torch.save(t, dst)
