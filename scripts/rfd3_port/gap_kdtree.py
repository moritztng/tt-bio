#!/usr/bin/env python3
"""RFD3_ATTN_KDTREE against the shipped row-blocked cdist index (host only, no device).

    gap_kdtree.py [OUT.json] [--threads=2,8] [--dump=FILE]

Cases are gap_nbr_prune.py's compact chains at 6051 atoms, seeds 0/1, coordinate noise 0 to 160 A
(the diffusion schedule's range), with the production mask (32 sequence neighbours, k=128), or a
dump of a real step's mask/seq_idx/x. Per case: rows that differ from the shipped build, rows that
differ from a float64 direct-difference reference, and both build times.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                        # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
THREADS = [int(t) for t in opts.get("threads", "2,8").split(",")]
L = 6051


def chain_like(n, seed):
    g = torch.Generator().manual_seed(seed)
    steps = torch.randn((n + 8) // 9, 3, generator=g)
    ca = torch.cumsum(3.8 * steps / steps.norm(dim=1, keepdim=True), 0)
    ca = (ca - ca.mean(0)) * (22.0 / ca.std(0).mean())
    return ca.repeat_interleave(9, 0)[:n] + torch.randn(n, 3, generator=g) * 1.2


def f64_reference(mask, seq_idx, x, k, block=256):
    xd, n = x[0].double(), x.shape[1]
    out = torch.empty(1, n, k, dtype=torch.long)
    for r0 in range(0, n, block):
        r1 = min(r0 + block, n)
        d = (xd[r0:r1, None] - xd[None]).pow(2).sum(-1).sqrt().masked_fill_(mask[r0:r1], float("inf"))
        fill = torch.topk(d, k, dim=-1, largest=False).indices.flip(dims=[-1]).unsqueeze(0)
        s = seq_idx[r0:r1].unsqueeze(0)
        out[:, r0:r1] = torch.where((s == n).expand_as(fill), fill, s.expand_as(fill))
    return out.sort(-1).values


def best(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1e3 * sorted(ts)[n // 2]


if "dump" in opts:
    D = torch.load(opts["dump"])
    cases = [("dump", D["mask"], D["seq_idx"], D["x"], D["k"])]
else:
    tok = torch.arange(L) // 9
    mask, seq_idx = M._mask_and_seq_idx(tok, 32, 128, None, None, L)
    cases = []
    for seed in (0, 1):
        for sg in (0.0, 0.5, 2.0, 8.0, 32.0, 160.0):
            xs = chain_like(L, seed) + sg * torch.randn(L, 3, generator=torch.Generator().manual_seed(seed + 1))
            cases.append((f"s{seed} sigma {sg:g}", mask, seq_idx, xs.unsqueeze(0).float(), 128))
rows = []
for nt in THREADS:
    torch.set_num_threads(nt)
    for name, mask, seq_idx, x, k in cases:
        ref = M._neighbours_row_blocked(mask, seq_idx, x, k).sort(-1).values
        kd = M._neighbours_kdtree(mask, seq_idx, x, k)
        f64 = f64_reference(mask, seq_idx, x, k)
        r = dict(threads=nt, case=name, rows_ne_shipped=int((kd != ref).any(-1).sum()),
                 rows_ne_f64=int((kd != f64).any(-1).sum()), shipped_rows_ne_f64=int((ref != f64).any(-1).sum()),
                 shipped_ms=best(lambda: M._neighbours_row_blocked(mask, seq_idx, x, k).sort(-1)),
                 kdtree_ms=best(lambda: M._neighbours_kdtree(mask, seq_idx, x, k)))
        rows.append(r)
        print("threads %2d %-16s kd!=shipped %4d  kd!=f64 %4d  shipped!=f64 %4d  shipped %7.1f ms  kd %6.1f ms  %.2fx"
              % (nt, name, r["rows_ne_shipped"], r["rows_ne_f64"], r["shipped_rows_ne_f64"], r["shipped_ms"],
                 r["kdtree_ms"], r["shipped_ms"] / r["kdtree_ms"]), flush=True)
if args:
    json.dump(dict(host=os.uname().nodename, L=L, rows=rows), open(args[0], "w"), indent=1)
