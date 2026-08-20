#!/usr/bin/env python3
"""RF3's confidence head does not layer-norm along the feature dimension.

With layer_norm_along_feature_dimension=False (this checkpoint) it calls

    F.layer_norm(S_trunk_I, normalized_shape=(S_trunk_I.shape))

i.e. normalized_shape is the WHOLE tensor shape, so the statistics are global scalars
over every element, not per-position over the last axis. A port that pattern-matches
the name and uses a normal last-dim layer norm gets a systematically different scale
and offset -- the exact error class PCC is nearly blind to (see the distogram head,
where a factor of 2 cost 7e-5 of pcc).

This quantifies the difference on the real captured tensors.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_conf.pt")
    args = ap.parse_args()
    cap = torch.load(args.capture, weights_only=False)

    rep = {}
    names = ["S_inputs_I", "S_trunk_I", "Z_trunk_II"]
    for i, name in enumerate(names):
        x = cap["in"][i]
        if not isinstance(x, torch.Tensor):
            continue
        x = x.float()
        whole = F.layer_norm(x, normalized_shape=tuple(x.shape))    # what RF3 does
        feat = F.layer_norm(x, normalized_shape=(x.shape[-1],))     # the natural mistake
        rep[name] = {
            "shape": list(x.shape),
            "whole_tensor_std": round(float(whole.std()), 6),
            "feature_wise_std": round(float(feat.std()), 6),
            "pcc_between_them": round(pcc(whole, feat), 6),
            "max_abs_diff": round(float((whole - feat).abs().max()), 4),
        }
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
