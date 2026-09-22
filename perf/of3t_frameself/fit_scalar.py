#!/usr/bin/env python3
"""of3t-frameself step 0: how much of the injected arm's disagreement is one scalar?

of3t-twoside's injected float64 arm overshoots `grads_f64_043.pt`'s trunk section by a norm
ratio near 1.75 at 46 of 48 blocks. Forty-eight independently-parameterised blocks do not
overshoot by the same factor through 48 arithmetic paths, so the candidate is a single scalar
applied once at the entry. This measures that directly: the best scalar a* = argmin_a
||a*g_inj - g_ref|| and the residual it leaves, as a fraction of ||g_ref||, pooled and per block.

  residual ~ 1e-12   it is EXACTLY a scalar and the question is only which one.
  residual large     it is not a scalar; the near-constant ratio is an artefact of averaging.

Reference is `grads_f64_043.pt`, the full-model float64 backward (R133: named on every line).
The arm is INJECTED, so it is the numerator and the bias runs from the injection.

    fit_scalar.py --ref grads_f64_043.pt --arm NAME=path.pt [...] --out FIT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import sys
from pathlib import Path

import torch

PRE = "pairformer_stack."


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fit(arm, ref, names):
    """<arm,ref>, ||arm||^2, ||ref||^2, ||arm-ref||^2 accumulated over a name set."""
    dot = arm_sq = ref_sq = err_sq = 0.0
    worst, worst_name, n_absent, n_bit = -1.0, None, 0, 0
    for n in names:
        r = ref[n]
        if r is None:
            continue
        r = r.to(torch.float64).reshape(-1)
        rn2 = float(torch.dot(r, r))
        ref_sq += rn2
        m = arm.get(n)
        if m is None:
            n_absent += 1
            err_sq += rn2
            continue
        m = m.to(torch.float64).reshape(-1)
        if torch.equal(m, r):
            n_bit += 1
        dot += float(torch.dot(m, r))
        arm_sq += float(torch.dot(m, m))
        e = float(torch.dot(m - r, m - r))
        err_sq += e
        rel = (e / rn2) ** 0.5 if rn2 > 0 else (0.0 if e == 0 else float("inf"))
        if rel > worst:
            worst, worst_name = rel, n
    a_star = dot / arm_sq if arm_sq > 0 else None
    res_sq = max(ref_sq - (dot ** 2 / arm_sq if arm_sq > 0 else 0.0), 0.0)
    return {
        "n_tensors": len(names), "n_absent_from_arm": n_absent, "n_bit_identical": n_bit,
        "rel_l2_as_is": (err_sq / ref_sq) ** 0.5 if ref_sq else None,
        "norm_ratio_arm_over_ref": (arm_sq / ref_sq) ** 0.5 if ref_sq else None,
        "cos": dot / (arm_sq * ref_sq) ** 0.5 if arm_sq > 0 and ref_sq > 0 else None,
        "best_scalar_a_star": a_star,
        "one_over_a_star": (1.0 / a_star) if a_star else None,
        "residual_after_best_scalar_frac_of_ref": (res_sq / ref_sq) ** 0.5 if ref_sq else None,
        "ref_squared_norm": ref_sq, "arm_squared_norm": arm_sq,
        "worst_rel_l2": worst if worst >= 0 else None, "worst_tensor": worst_name,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ref_raw = torch.load(a.ref, map_location="cpu", weights_only=False, mmap=True)
    names = sorted(k for k in ref_raw if k.startswith(PRE) and ref_raw[k] is not None)
    print(f"reference {a.ref}: {len(names)} trunk tensors", flush=True)
    ref = {n: ref_raw[n] for n in names}

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
        "reference": {"path": str(a.ref), "sha256": sha256_file(a.ref),
                      "is": "the full-model float64 backward on batch_step003, num_recycles 0"},
        "n_trunk_tensors": len(names),
        "arms": {},
    }
    for spec in a.arm:
        nm, p = spec.split("=", 1)
        print(f"arm {nm} {p}", flush=True)
        d = torch.load(p, map_location="cpu", weights_only=False)
        g = d["grads"] if isinstance(d, dict) and "grads" in d else d
        rec = {"path": p, "sha256": sha256_file(p), "bytes": os.path.getsize(p),
               "recorded_loss": d.get("loss") if isinstance(d, dict) else None,
               "recorded_policy": d.get("policy") if isinstance(d, dict) else None,
               "recorded_tree": d.get("tree") if isinstance(d, dict) else None,
               "injected_side": "arm (numerator). The reference is not injected."}
        rec["pooled"] = fit(g, ref, names)
        print("  pooled " + json.dumps({k: rec["pooled"][k] for k in
              ("rel_l2_as_is", "norm_ratio_arm_over_ref", "cos", "best_scalar_a_star",
               "residual_after_best_scalar_frac_of_ref")}), flush=True)
        pb = {}
        for i in range(a.blocks):
            bn = [n for n in names if n.startswith(f"{PRE}blocks.{i}.")]
            if bn:
                pb[str(i)] = fit(g, ref, bn)
        rec["per_block"] = pb
        rr = [v["norm_ratio_arm_over_ref"] for v in pb.values()]
        aa = [v["best_scalar_a_star"] for v in pb.values()]
        res = [v["residual_after_best_scalar_frac_of_ref"] for v in pb.values()]
        cc = [v["cos"] for v in pb.values()]
        def summ(xs):
            return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
                    "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                    "min": min(xs), "max": max(xs),
                    "cv_pct": 100.0 * statistics.stdev(xs) / statistics.fmean(xs)
                    if len(xs) > 1 and statistics.fmean(xs) else None}
        rec["per_block_summary"] = {
            "norm_ratio": summ(rr), "best_scalar_a_star": summ(aa),
            "residual_after_best_scalar": summ(res), "cos": summ(cc),
            "blocks_outside_ratio_1_6_to_1_9": sorted(
                (int(k), v["norm_ratio_arm_over_ref"]) for k, v in pb.items()
                if not 1.6 <= v["norm_ratio_arm_over_ref"] <= 1.9),
        }
        # The pooled best scalar applied to every block: does ONE number fit all 48, or do 48
        # different numbers each fit their own block? The second is not "a single scalar at the
        # entry", it is 48 coincidences.
        ap_ = rec["pooled"]["best_scalar_a_star"]
        rec["pooled_scalar_applied_per_block"] = {
            k: math.sqrt(max(v["ref_squared_norm"] - 2 * ap_ * (v["cos"] * math.sqrt(
                v["arm_squared_norm"] * v["ref_squared_norm"])) + ap_ ** 2 * v["arm_squared_norm"],
                0.0) / v["ref_squared_norm"]) for k, v in pb.items()}
        rec["verdict_step0"] = (
            "EXACTLY A SCALAR" if rec["pooled"]["residual_after_best_scalar_frac_of_ref"] <= 1e-12
            else "NOT A SCALAR: %.6g of the reference norm survives the best single scalar"
            % rec["pooled"]["residual_after_best_scalar_frac_of_ref"])
        print("  " + rec["verdict_step0"], flush=True)
        out["arms"][nm] = rec
        del d, g

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
