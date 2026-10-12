#!/usr/bin/env python3
"""Exact spatial prune for RFD3's atom neighbour index, screened against the shipped build (CPU only).

    gap_nbr_prune.py [THREADS] [--dump=attn_inputs.pt] [--atoms=6051] [--cand=1024] [--block=256]

The shipped `_neighbours_row_blocked` runs cdist + masked_fill + top-k over every key of every row: ~0.5 core-s a call
at 6051 atoms. Every row is independent, so rows can be visited in any order. This one sorts them into spatially
compact blocks (Morton order of the coordinates), takes for each block the `cand` keys nearest its centroid c (kept in
index order, so ties sort as in the full row) and runs the same chain over those. Any key outside that set is at least
R - rb from every row of the block (R: the set's largest distance to c, rb: the block's radius), so a row whose k-th
distance in the set is below R - rb has the full row's top-k; every other row is recomputed over the full row.
The check is torch.equal on the output, at diffusion-like noise levels on a chain-like structure, or on a dump.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                        # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
THREADS = [int(t) for t in (args[0] if args else "1,16").split(",")]
L = int(opts.get("atoms", 6051))
CAND = int(opts.get("cand", 1024))
BLOCK = int(opts.get("block", 256))


def morton(x, bits=10):
    """Morton code of each row of x [L, 3]."""
    lo, hi = x.min(0).values, x.max(0).values
    q = ((x - lo) / (hi - lo).clamp(min=1e-12) * (2 ** bits - 1)).long()
    code = torch.zeros(x.shape[0], dtype=torch.long)
    for b in range(bits):
        for a in range(3):
            code |= ((q[:, a] >> b) & 1) << (3 * b + a)
    return code


def neighbours_pruned(mask, seq_idx, x, k, block=BLOCK, cand=CAND):
    length = x.shape[1]
    k = min(k, length)
    if length <= 2 * cand:
        return M._neighbours_row_blocked(mask, seq_idx, x, k), 0
    xs = x[0]
    inf = torch.tensor(float("inf"), dtype=x.dtype)
    fill = torch.empty(length, k, dtype=torch.long)
    order = torch.argsort(morton(xs))
    redo = []
    for r0 in range(0, length, block):
        rows = order[r0:r0 + block]
        q = xs[rows]
        c = q.mean(0)
        rb = (q - c).norm(dim=1).max()
        dc = (xs - c).norm(dim=1)
        dsel, keys = torch.topk(dc, cand, largest=False)
        keys = keys.sort().values
        d = torch.cdist(q.unsqueeze(0), xs[keys].unsqueeze(0), p=2)[0].masked_fill_(mask[rows][:, keys], inf)
        val, pos = torch.topk(d, k, dim=-1, largest=False)
        ok = val[:, -1] < dsel.max() - rb
        fill[rows[ok]] = keys[pos[ok]].flip(dims=[-1])
        redo.append(rows[~ok])
    redo = torch.cat(redo).sort().values
    for r0 in range(0, len(redo), block):
        rows = redo[r0:r0 + block]
        d = torch.cdist(xs[rows].unsqueeze(0), x, p=2)[0].masked_fill_(mask[rows], inf)
        fill[rows] = torch.topk(d, k, dim=-1, largest=False).indices.flip(dims=[-1])
    s = seq_idx.unsqueeze(0)
    out = torch.where((s == length).expand(1, length, k), fill.unsqueeze(0), s.expand(1, length, k))
    return out, len(redo)


def chain_like(n, seed):
    """A compact chain: residues on a self-avoiding-ish random walk, ~9 atoms each within 2 A."""
    g = torch.Generator().manual_seed(seed)
    n_res = (n + 8) // 9
    steps = torch.randn(n_res, 3, generator=g)
    steps = 3.8 * steps / steps.norm(dim=1, keepdim=True)
    ca = torch.cumsum(steps, 0)
    ca = (ca - ca.mean(0)) * (22.0 / ca.std(0).mean())       # squeeze into a ~globule
    atoms = ca.repeat_interleave(9, 0)[:n] + torch.randn(n, 3, generator=g) * 1.2
    return atoms


def inputs(seed, sigma):
    xs = chain_like(L, seed)
    xs = xs + sigma * torch.randn(L, 3, generator=torch.Generator().manual_seed(seed + 1))
    tok = torch.arange(L) // 9
    mask, seq_idx = M._mask_and_seq_idx(tok, 32, 128, None, None, L)
    return mask, seq_idx, xs.unsqueeze(0).float()


if "dump" in opts:
    D = torch.load(opts["dump"])
    cases = [("dump", D["mask"], D["seq_idx"], D["x"], D["k"])]
else:
    cases = [("s%d sigma %g" % (seed, sg),) + inputs(seed, sg) + (128,)
             for seed in (0, 1) for sg in (0.0, 0.5, 2.0, 8.0, 32.0, 160.0)]
for nt in THREADS:
    torch.set_num_threads(nt)
    for name, mask, seq_idx, x, k in cases:
        ref = M._neighbours_row_blocked(mask, seq_idx, x, k)
        new, nredo = neighbours_pruned(mask, seq_idx, x, k)
        t = {}
        for arm, fn in (("ref", lambda: M._neighbours_row_blocked(mask, seq_idx, x, k)),
                        ("prune", lambda: neighbours_pruned(mask, seq_idx, x, k))):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            t[arm] = 1e3 * sorted(ts)[1]
        print("threads %2d %-18s equal %-5s redo %4d/%d  ref %7.1f ms  prune %7.1f ms  %.2fx"
              % (nt, name, torch.equal(ref, new), nredo, L, t["ref"], t["prune"], t["ref"] / t["prune"]), flush=True)
