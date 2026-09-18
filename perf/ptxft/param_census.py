"""Parameter census of the real Protenix v2 checkpoint, by module.

Phase 0 arithmetic for the LoRA-vs-full decision. Host only, no device: this reads
the checkpoint torch.load gives us and counts elements, so the numbers are the
shipped weights and not a config file's idea of them.
"""
from __future__ import annotations

import argparse
import collections
import os

import torch


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "ema", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--depth", type=int, default=2)
    a = ap.parse_args()
    sd = load(a.ckpt)
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    total = sum(v.numel() for v in tensors.values())
    print(f"checkpoint {a.ckpt}")
    print(f"tensors {len(tensors)}  params {total:,}  ({total*4/1e9:.3f} GB fp32, {total*2/1e9:.3f} GB bf16)")

    groups = collections.Counter()
    counts = collections.Counter()
    for k, v in tensors.items():
        parts = k.split(".")
        groups[".".join(parts[:a.depth])] += v.numel()
        counts[".".join(parts[:a.depth])] += 1
    print()
    print(f"{"prefix":<52} {"params":>14} {"pct":>6} {"n":>5}")
    for g, n in groups.most_common(40):
        print(f"{g:<52} {n:>14,} {100.0*n/total:>5.1f}% {counts[g]:>5}")

    # Linear-shaped 2D weights, which is what a LoRA adapter can touch.
    lin = {k: v for k, v in tensors.items() if v.dim() == 2 and k.endswith("weight")}
    lin_n = sum(v.numel() for v in lin.values())
    print()
    print(f"2D .weight tensors (LoRA-adaptable): {len(lin)} tensors, {lin_n:,} params "
          f"({100.0*lin_n/total:.1f}%)")

    # Per-block trunk census: how much lives in the pairformer stack.
    for prefix in ("pairformer_stack", "pairformer", "trunk"):
        blk = collections.Counter()
        for k, v in tensors.items():
            if prefix in k:
                blk[prefix] += v.numel()
        if blk:
            print(f"{prefix}: {blk[prefix]:,} ({100.0*blk[prefix]/total:.1f}%)")


if __name__ == "__main__":
    main()
