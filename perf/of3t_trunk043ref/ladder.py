#!/usr/bin/env python3
"""Depth ladder: how our two tracks' disagreement with the 0.4.3 reference grows with block count.

A 48-block composed reading cannot say whether the error is made once per block and accumulates
or is made once and carried. `of3t-trunkfwd` answered that against a 0.5.0 reference and got
k^0.925 shipped -- nearly coherent, the signature of a systematic per-block bias -- against
k^0.584 with the bias orientation flipped, which is a random walk. Against the reference the
checkpoint is actually bound to, the same question has a different answer and this measures it.

Both tracks. Our port and upstream 0.4.3 both run k blocks from the SAME captured block-0 input,
so at every rung the two sides compute the same function of the same input and only the
arithmetic differs. The exponent is fitted on the masked relative L2 over the rungs.

One process per side per rung: ttnn and an upstream release tree cannot share an interpreter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def metrics(ours, ref):
    a = ours.to(torch.float64).reshape(-1)
    b = ref.to(torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    den = nb if nb > 0 else 1e-300
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / den),
            "norm_ratio": na / den, "cos": float((a @ b) / ((na * nb) or 1e-300))}


def fit(ks, ys):
    """log y = log c + p log k, least squares on the rungs with y > 0."""
    pts = [(math.log(k), math.log(y)) for k, y in zip(ks, ys) if y > 0 and k > 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rungs", default="1,2,4,8,16,24,32,48")
    a = ap.parse_args()

    ks = [int(x) for x in a.rungs.split(",")]
    os.makedirs(a.work, exist_ok=True)
    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(sm.shape[1])
    msk = sm.reshape(1, N, 1)
    pmk = pm.reshape(1, N, N, 1)

    rows = []
    for k in ks:
        ref = os.path.join(a.work, f"ladder_ref043_k{k}.pt")
        dev = os.path.join(a.work, f"ladder_dev_k{k}")
        if not os.path.isfile(ref):
            subprocess.run([a.python, os.path.join(HERE, "ref_stack.py"),
                            "--tree", a.tree, "--boundary", a.boundary, "--policy", "f64",
                            "--blocks", str(k), "--out", ref,
                            "--report", ref + ".json"], check=True,
                           stdout=subprocess.DEVNULL)
        if not os.path.isfile(os.path.join(dev, "device_shipped.pt")):
            env = dict(os.environ, TT_VISIBLE_DEVICES="0", TT_BIO_LEASE_CARDS="0",
                       TT_BIO_LEASE_HOLDER="worker:of3t-trunk043ref")
            subprocess.run([a.python, os.path.join(HERE, "device_arm.py"),
                            "--boundary", a.boundary, "--blocks", str(k),
                            "--outdir", dev, "--report", os.path.join(dev, "report.json")],
                           check=True, env=env, stdout=subprocess.DEVNULL)
        r = torch.load(ref, map_location="cpu", weights_only=False)
        for arm in ("shipped", "lever"):
            d = torch.load(os.path.join(dev, f"device_{arm}.pt"), map_location="cpu",
                           weights_only=False)
            rows.append({"k": k, "arm": arm,
                         "s_masked": metrics(d["s"] * msk, r["s"] * msk),
                         "z_masked": metrics(d["z"] * pmk, r["z"] * pmk)})
            print(f"k={k:<3} {arm:<8} s {rows[-1]['s_masked']['rel_l2']:.6e} "
                  f"(r {rows[-1]['s_masked']['norm_ratio']:.6f})   "
                  f"z {rows[-1]['z_masked']['rel_l2']:.6e} "
                  f"(r {rows[-1]['z_masked']['norm_ratio']:.6f})", flush=True)

    rep = {"what": __doc__.strip().splitlines()[0], "tokens": N, "real_tokens": int(sm.sum()),
           "rungs": ks, "rows": rows, "exponents": {}}
    for arm in ("shipped", "lever"):
        for tr in ("s_masked", "z_masked"):
            sel = [(x["k"], x[tr]["rel_l2"]) for x in rows if x["arm"] == arm]
            rep["exponents"][f"{arm}.{tr}"] = fit([k for k, _ in sel], [y for _, y in sel])
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep["exponents"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
