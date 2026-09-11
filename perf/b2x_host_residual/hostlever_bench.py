#!/usr/bin/env python3
"""Bit-exact host levers for the two biggest un-ported stages, at the real fold shapes.

Everything in `DiffusionConditioning` is a per-position operation on a b*n*n*C tensor. At 512
tokens that tensor is 128 MB and the stage walks it a dozen times, so most of its time is DRAM
traffic, not arithmetic. Row-blocking changes only the loop order: the layer-norm statistics are
per-row and every matmul keeps its K, so nothing's reduction changes length or order and the
result has to be bit-identical. That is checked with `torch.equal`, not a tolerance.

  transition   `Transition.forward` on z: norm, fc1+fc2 to 2C, SiLU-gate, fc3 back to C. The
               two 2C intermediates are 268 MB each at 512 tokens.
  pairwise     `PairwiseConditioning.forward`: cat(z_trunk, relpos) -> 268 MB, LayerNorm(2C) +
               Linear(2C -> C), then two residual Transitions. Blocking also removes the cat.
  bias         the 24 (LayerNorm(C) + Linear(C, H)) of `token_trans_proj_z` over z, and the
               3 + 3 of `atom_enc_proj_z` / `atom_dec_proj_z` over p.
"""
import argparse
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.environ.get("B2X_REPO", "/home/moritz/tt-bio"))
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
from tt_bio.boltz2 import PairwiseConditioning, Transition  # noqa: E402


def timeit(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def blocked_transition(tr, x, rows):
    c = x.shape[-1]
    flat = x.reshape(-1, c)
    out = flat.new_empty(flat.shape)
    for s in range(0, flat.shape[0], rows):
        out[s:s + rows] = tr(flat[s:s + rows])
    return out.reshape(x.shape)


def blocked_pairwise(pc, z_trunk, relpos, rows):
    """cat + init proj + the two residual transitions, one row block at a time."""
    c = z_trunk.shape[-1]
    zt = z_trunk.reshape(-1, c)
    rp = relpos.reshape(-1, c)
    out = zt.new_empty(zt.shape)
    for s in range(0, zt.shape[0], rows):
        blk = pc.dim_pairwise_init_proj(torch.cat((zt[s:s + rows], rp[s:s + rows]), dim=-1))
        for tr in pc.transitions:
            blk = tr(blk) + blk
        out[s:s + rows] = blk
    return out.reshape(z_trunk.shape)


def blocked_bias(layers, x, rows):
    c = x.shape[-1]
    h = layers[0][1].out_features
    flat = x.reshape(-1, c)
    out = flat.new_empty(flat.shape[0], h * len(layers))
    for s in range(0, flat.shape[0], rows):
        blk = flat[s:s + rows]
        for i, layer in enumerate(layers):
            out[s:s + rows, i * h:(i + 1) * h] = layer(blk)
    return out.reshape(*x.shape[:-1], h * len(layers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--token-z", type=int, default=128)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--rows", type=int, nargs="+", default=[2048, 8192, 32768])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    n, c = a.n, a.token_z
    print(f"host {os.uname().nodename}  torch {torch.__version__}  "
          f"threads {torch.get_num_threads()}  n={n} C={c}")

    torch.manual_seed(5)
    z = torch.randn(1, n, n, c)
    relpos = torch.randn(1, n, n, c)

    # --- Transition alone -------------------------------------------------------------------
    tr = Transition(dim=c, hidden=2 * c).eval()
    ref = tr(z)
    t_ship = timeit(lambda: tr(z), a.reps)
    print(f"\ntransition   shipped {1e3*t_ship:8.2f} ms")
    for rows in a.rows:
        got = blocked_transition(tr, z, rows)
        t = timeit(lambda: blocked_transition(tr, z, rows), a.reps)
        print(f"             rows={rows:7d} {1e3*t:8.2f} ms  {t_ship/t:5.2f}x  "
              f"bit-exact {torch.equal(ref, got)}")
        del got
    del ref

    # --- PairwiseConditioning ---------------------------------------------------------------
    pc = PairwiseConditioning(token_z=c, dim_token_rel_pos_feats=c, num_transitions=2).eval()
    ref = pc(z, relpos)
    t_ship = timeit(lambda: pc(z, relpos), a.reps)
    print(f"\npairwise     shipped {1e3*t_ship:8.2f} ms   (cat is {n*n*2*c*4/2**20:.0f} MB)")
    for rows in a.rows:
        got = blocked_pairwise(pc, z, relpos, rows)
        t = timeit(lambda: blocked_pairwise(pc, z, relpos, rows), a.reps)
        print(f"             rows={rows:7d} {1e3*t:8.2f} ms  {t_ship/t:5.2f}x  "
              f"bit-exact {torch.equal(ref, got)}")
        del got
    del ref

    # --- the 24-layer bias stack ------------------------------------------------------------
    torch.manual_seed(6)
    layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, a.heads, bias=False))
                            for _ in range(a.layers)]).eval()
    ref = torch.cat([l(z) for l in layers], dim=-1)
    t_ship = timeit(lambda: torch.cat([l(z) for l in layers], dim=-1), a.reps)
    print(f"\nbias {a.layers}x{a.heads}  shipped {1e3*t_ship:8.2f} ms   "
          f"(out is {n*n*a.heads*a.layers*4/2**20:.0f} MB)")
    for rows in a.rows:
        got = blocked_bias(layers, z, rows)
        t = timeit(lambda: blocked_bias(layers, z, rows), a.reps)
        print(f"             rows={rows:7d} {1e3*t:8.2f} ms  {t_ship/t:5.2f}x  "
              f"bit-exact {torch.equal(ref, got)}")
        del got
    return 0


if __name__ == "__main__":
    sys.exit(main())
