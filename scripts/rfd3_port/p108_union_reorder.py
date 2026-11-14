#!/usr/bin/env python3
"""p108 -- block key unions of the atom attention index in sequence order vs a spatial order.

block_sparse.py is off by default because it is target-specific: Q has to divide the padded atom
axis, and Q had to be large (1216) because at Q = 320 the WIDEST block union was still 2080 of 6080
keys at the median step (p102), so the gathered rows nb * U grew with the block count. Neither is a
property of the index. Atoms in sequence order put a fold's spatial neighbours at distant sequence
positions; sorted along a space-filling curve of the very coordinates the k-NN index is built from,
32 consecutive atoms are spatial neighbours, so their 128-neighbour sets mostly coincide. If the
unions shrink enough, Q = 32 divides every padded axis and the arm stops being target-specific.

This spies on `_create_attention_indices` over a full design and, for every atom index, records
the block union width (max and mean over blocks) per Q in sequence order and in Morton order of
the coordinates the index was built from. Scalars only. Arch-independent: the index depends on
the coordinates, not on which chip ran the step.

    python scripts/rfd3_port/p108_union_reorder.py OUT.json [STEPS] [SEED] [FIXTURE] [CKPT]
"""
import json
import os
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p108/union_reorder.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FIXTURE = pathlib.Path(sys.argv[4] if len(sys.argv) > 4 else "perf/dsfix/fixtures/rfd3_R4.json")
CKPT = sys.argv[5] if len(sys.argv) > 5 else os.path.expanduser("~/.boltz/rfd3/weights")
QBLOCKS = (32, 64, 128, 320, 608, 1216)
K = 128
CHUNKS = ((32, 32), (64, 64), (128, 128), (256, 256), (128, 512))   # (query chunk, key chunk) an SDPA could skip at

_orig = M._create_attention_indices
SEEN = {"n": 0}
ROWS = []


def morton(x):
    """Permutation sorting [L, 3] points along a 10-bit-per-axis Morton curve of their bounding box."""
    lo = x.min(0).values
    span = (x.max(0).values - lo).clamp_min(1e-6)
    q = ((x - lo) / span * 1023).round().long().clamp(0, 1023)
    code = torch.zeros(x.shape[0], dtype=torch.long)
    for b in range(10):
        for a in range(3):
            code |= ((q[:, a] >> b) & 1) << (3 * b + a)
    return torch.argsort(code)


def unions(idx, n_key):
    """{Q: (max, mean)} block union widths of an [L, K] index whose rows are already in block order."""
    L = idx.shape[0]
    rows = (L + 31) // 32 * 32
    pad = torch.cat([idx, idx[-1:].expand(rows - L, idx.shape[1])], 0)
    out = {}
    for Q in QBLOCKS:
        if rows % Q:
            continue
        nb = rows // Q
        mask = torch.zeros(nb, n_key, dtype=torch.bool)
        mask.scatter_(1, pad.reshape(nb, Q * idx.shape[1]), True)
        w = mask.sum(1).float()
        out[Q] = (int(w.max()), round(float(w.mean()), 1))
    return out


def density(idx, n_key):
    """{"Qc/Kc": mean fraction of key chunks a query chunk touches} for an [L, K] index whose rows and
    key ids are both in the same order: the share of score tiles a block-skipping SDPA still computes."""
    L = idx.shape[0]
    out = {}
    for qc, kc in CHUNKS:
        rows = (L + qc - 1) // qc * qc
        pad = torch.cat([idx, idx[-1:].expand(rows - L, idx.shape[1])], 0)
        nk = (n_key + kc - 1) // kc
        hit = torch.zeros(rows // qc, nk, dtype=torch.bool)
        hit.scatter_(1, (pad // kc).reshape(rows // qc, qc * idx.shape[1]), True)
        out["%d/%d" % (qc, kc)] = round(float(hit.float().mean()), 4)
    return out


def _spy(f, X_L, tok_idx, n_keys, n_seq_neighbours, **kw):
    idx = _orig(f, X_L, tok_idx, n_keys, n_seq_neighbours, **kw)
    SEEN["n"] += 1
    if idx.shape[-1] == K and idx.shape[-2] > 2000:
        i = idx[0].detach().cpu().long()
        x = X_L.detach().cpu().float().reshape(-1, X_L.shape[-2], 3)[0]
        n_key = (i.shape[0] + 31) // 32 * 32
        p = morton(x)
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.numel())
        ROWS.append(dict(call_no=SEEN["n"], seq=unions(i, n_key), morton=unions(inv[i[p]], n_key),
                         dens_seq=density(i, n_key), dens_morton=density(inv[i[p]], n_key)))
    return idx


M._create_attention_indices = _spy


def main():
    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p108_%d" % os.getpid()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    print("[p108] atom indices %d of %d index builds" % (len(ROWS), SEEN["n"]), flush=True)
    rep = {}
    for order in ("seq", "morton"):
        for Q in QBLOCKS:
            mx = [r[order][Q][0] for r in ROWS if Q in r[order]]
            mn = [r[order][Q][1] for r in ROWS if Q in r[order]]
            if not mx:
                continue
            s = sorted(mx)
            rep["%s/Q%d" % (order, Q)] = dict(
                umax_p50=s[len(s) // 2], umax_p90=s[int(0.9 * (len(s) - 1))], umax_max=s[-1],
                umean_p50=statistics.median(mn))
            print("[p108] %-6s Q %4d  u_max p50 %5d p90 %5d max %5d  u_mean p50 %7.1f"
                  % (order, Q, s[len(s) // 2], s[int(0.9 * (len(s) - 1))], s[-1], statistics.median(mn)),
                  flush=True)
    for order in ("seq", "morton"):
        for c in ROWS[0]["dens_" + order]:
            d = sorted(r["dens_" + order][c] for r in ROWS)
            rep["dens_%s/%s" % (order, c)] = dict(mean=round(statistics.mean(d), 4), p50=d[len(d) // 2],
                                                  p90=d[int(0.9 * (len(d) - 1))], max=d[-1])
            print("[p108] tile density %-6s q/k %-7s mean %.3f p50 %.3f p90 %.3f max %.3f"
                  % (order, c, statistics.mean(d), d[len(d) // 2], d[int(0.9 * (len(d) - 1))], d[-1]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(steps=STEPS, seed=SEED, fixture=str(FIXTURE), rows=ROWS, report=rep), indent=1))
    print("[p108] wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
