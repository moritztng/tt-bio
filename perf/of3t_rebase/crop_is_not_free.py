#!/usr/bin/env python3
"""Is the 64-token crop free on the REFERENCE side? Reference against itself, no device.

`instrument_a_bundle.py --crop 64` exists on a stated premise, in its own help text: "The batch
is 384 tokens of which 56 are real, and a layer-norm WEIGHT gradient sums over every position
including the 328 padded ones. **Their block zeroes those through single_mask/pair_mask**; if
ours does not, the forward can agree on the real tokens while the weight gradient does not."

The D28 NaN probe refutes the bolded clause. Poisoning the 8 pad rows of a crop-64 boundary with
NaN makes **57 of 57** of their own parameter gradients NaN. Their block does not zero the pad
out of the weight-gradient path; it carries it.

If the pad contributes on their side too, then cropping 384 to 64 deletes 320 contributing
positions and changes the reference function. Every crop-64 arm compares our 64-token backward
against the bundle's 384-token gradient, so any difference the crop makes is baked into those
numbers as an offset nobody subtracted.

This measures that offset directly, with no device and no port involved: their block, their
weights, their captured boundary, differentiated at 384 and at 64, compared per parameter. A
small number vindicates every crop-64 arm on the branch. A large one says the ladder was
measuring the crop as well as the port.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))

BAR = 5.0e-02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=refpath.CAP)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_rebase/crop_is_not_free.json"))
    a = ap.parse_args()

    from instrument_a_stack import their_stack, load_ckpt

    cap = torch.load(f"{a.cap}/block{a.block}_boundary.pt", map_location="cpu",
                     weights_only=False)
    args_in = cap["args"]
    s0, z0 = args_in[0].to(torch.float64), args_in[1].to(torch.float64)
    sm = args_in[2].to(torch.float64) if len(args_in) > 2 else None
    pm = args_in[3].to(torch.float64) if len(args_in) > 3 else None
    for k, v in (cap["kwargs"] or {}).items():
        if torch.is_tensor(v):
            if "single" in k:
                sm = v.to(torch.float64)
            elif "pair" in k or "mask" in k:
                pm = v.to(torch.float64)
    cot_s, cot_z = cap["cot"][0].to(torch.float64), cap["cot"][1].to(torch.float64)

    sd = load_ckpt()

    def grads_at(c):
        blk = their_stack(sd, 1, first=a.block)[0][0]
        blk.eval()
        for q in blk.parameters():
            q.grad = None
        S, Z = (s0, z0) if c is None else (s0[:, :c].contiguous(), z0[:, :c, :c].contiguous())
        M = sm if (sm is None or c is None) else sm[:, :c].contiguous()
        P = pm if (pm is None or c is None) else pm[:, :c, :c].contiguous()
        CS = cot_s if c is None else cot_s[:, :c].contiguous()
        CZ = cot_z if c is None else cot_z[:, :c, :c].contiguous()
        sr, zr = blk(S, Z, M, P)
        ((sr * CS).sum() + (zr * CZ).sum()).backward()
        g = {n: (q.grad.detach().clone() if q.grad is not None else None)
             for n, q in blk.named_parameters()}
        n_tok = int(Z.shape[1])
        real = None if M is None else int(M.sum())
        del blk, sr, zr
        return g, n_tok, real

    g_full, n_full, real_full = grads_at(None)
    print(f"full: {n_full} tokens, {real_full} real", flush=True)
    g_crop, n_crop, real_crop = grads_at(a.crop)
    print(f"crop: {n_crop} tokens, {real_crop} real", flush=True)

    rows = []
    for k in sorted(set(g_full) & set(g_crop)):
        vf, vc = g_full[k], g_crop[k]
        if vf is None or vc is None:
            continue
        nf = float(torch.linalg.vector_norm(vf))
        rows.append({"param": k, "ref_norm": nf,
                     "rel_l2": float(torch.linalg.vector_norm(vc - vf) / (nf + 1e-300))})
    kept = [r for r in rows if r["ref_norm"] >= 1e-12]
    v = [r["rel_l2"] for r in kept]
    worst = max(kept, key=lambda r: r["rel_l2"])
    sq = sum(r["ref_norm"] ** 2 for r in kept) or 1.0
    over = [r for r in kept if r["rel_l2"] > BAR]
    rep = {
        "instrument": "the reference block differentiated at full length and at the crop, "
                      "compared against itself",
        "premise_under_test": "instrument_a_bundle.py --crop help: 'Their block zeroes those "
                              "[padded positions] through single_mask/pair_mask'",
        "block": a.block, "capture": a.cap,
        "full": {"tokens": n_full, "real": real_full},
        "crop": {"tokens": n_crop, "real": real_crop},
        "n_compared": len(kept), "n_dropped_by_a14": len(rows) - len(kept),
        "median_rel_l2": statistics.median(v) if v else None,
        "worst_rel_l2": worst["rel_l2"], "worst_param": worst["param"],
        "over_bar_5e-2": len(over),
        "over_bar_norm_share": sum(r["ref_norm"] ** 2 for r in over) / sq,
        "per_param": sorted(kept, key=lambda r: -r["rel_l2"]),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"crop {n_crop} vs full {n_full}, same weights, same boundary, reference both sides:")
    print(f"  median {rep['median_rel_l2']:.4e}   worst {rep['worst_rel_l2']:.4e} on "
          f"{rep['worst_param']}")
    print(f"  over the {BAR:.0e} bar: {len(over)} of {len(kept)}, holding "
          f"{100*rep['over_bar_norm_share']:.1f} % of the compared squared norm")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
