#!/usr/bin/env python3
"""p99 -- the host cost of the block-union index, which every device screen so far excluded.

p95/p96/p97/p98/p100 build the block index once, outside the timing loop, then time the device
chain. Production cannot do that: the atom neighbour index is rebuilt from the coordinates, and
p94 counted about one atom index per diffusion step, so the union and the block-local positions
are per-STEP host work, 200 times a design. Host cost in this model is additive, so whatever
this costs comes straight off any isolated device prize.

Two formulations of the same index, timed against each other and checked identical:

  unique   what p96 used: torch.unique per block (a sort) plus searchsorted per block
  bitmap   scatter the block's K neighbours into a [nb, NK] bool mask; the union is the mask's
           nonzero columns, and the block-local position of every neighbour is
           (mask.cumsum(1) - 1) gathered at the neighbour. No sort, no per-block python loop.

Reports ms/step and s/design at 200 steps.
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p99/index_host.json")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TAG = sys.argv[3] if len(sys.argv) > 3 else "early"
QSWEEP = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "160,320,608,1216,3040").split(",")]
IDX_PT = pathlib.Path("perf/p94/indices.pt")
L, NK, K, STEPS = 6051, 6080, 128, 200


def tile(n):
    return ((n + 31) // 32) * 32


def build_unique(idx_pad, nb, Q, U):
    gather = torch.zeros(nb, U, dtype=torch.int64)
    pos = torch.zeros(nb * Q, K, dtype=torch.int64)
    for b in range(nb):
        blk = idx_pad[b * Q:(b + 1) * Q]
        u = torch.unique(blk)
        gather[b, :u.numel()] = u
        pos[b * Q:(b + 1) * Q] = torch.searchsorted(u, blk)
    return gather, pos


def build_bitmap(idx_pad, nb, Q, U):
    blk = idx_pad.reshape(nb, Q * K)
    mask = torch.zeros(nb, NK, dtype=torch.bool)
    mask.scatter_(1, blk, True)
    rank = mask.cumsum(1) - 1                       # block-local column of every key
    pos = rank.gather(1, blk).reshape(nb * Q, K)
    gather = torch.zeros(nb, U, dtype=torch.int64)
    nz = mask.nonzero()                             # [total, 2] of (block, key), block-major
    gather[nz[:, 0], rank[nz[:, 0], nz[:, 1]]] = nz[:, 1]
    return gather, pos


def timeit(fn, n=REPS, warm=3):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def main():
    torch.set_num_threads(1)      # production builds this inline in the fold loop
    idx = torch.load(IDX_PT)[TAG].long()
    nb_rows = tile(L)
    idx_pad = torch.cat([idx, idx[-1:].expand(nb_rows - L, K)], 0)
    # U must be a compile-time constant or every step recompiles, so the bound that matters is
    # the widest union over every schedule point, not the one the 'early' index happens to give.
    all_tags = sorted(torch.load(IDX_PT).keys())
    tag_u = {}
    for t in all_tags:
        it = torch.load(IDX_PT)[t].long()
        ip = torch.cat([it, it[-1:].expand(nb_rows - it.shape[0], K)], 0)
        tag_u[t] = {}
        for Q in QSWEEP:
            nb = nb_rows // Q
            if nb * Q != nb_rows:
                continue
            tag_u[t][Q] = max(int(torch.unique(ip[b * Q:(b + 1) * Q]).numel()) for b in range(nb))
    for Q in QSWEEP:
        if Q not in tag_u[all_tags[0]]:
            continue
        per = {t: tag_u[t][Q] for t in all_tags}
        worst = max(per.values())
        print("Q=%4d  u_max per tag %s  -> bucket bound U=%d (the 'early' tag alone says %d)"
              % (Q, per, tile(worst), tile(per.get(TAG, worst))), flush=True)
    print("", flush=True)

    rows = []
    for Q in QSWEEP:
        nb = nb_rows // Q
        if nb * Q != nb_rows:
            print("[p99] Q=%d does not divide %d, skipped" % (Q, nb_rows), flush=True)
            continue
        u_max = max(int(torch.unique(idx_pad[b * Q:(b + 1) * Q]).numel()) for b in range(nb))
        U = tile(u_max)
        g_u, p_u = build_unique(idx_pad, nb, Q, U)
        g_b, p_b = build_bitmap(idx_pad, nb, Q, U)
        same_g = bool(torch.equal(g_u, g_b))
        same_p = bool(torch.equal(p_u, p_b))
        t_u = timeit(lambda: build_unique(idx_pad, nb, Q, U))
        t_b = timeit(lambda: build_bitmap(idx_pad, nb, Q, U))
        print("Q=%4d nb=%3d U=%4d  unique %8.3f ms/step -> %6.3f s/design   "
              "bitmap %8.3f ms/step -> %6.3f s/design   identical g=%s p=%s"
              % (Q, nb, U, t_u, t_u * STEPS / 1000.0, t_b, t_b * STEPS / 1000.0,
                 same_g, same_p), flush=True)
        rows.append(dict(Q=Q, n_blocks=nb, u_width=U, u_max=u_max,
                         unique_ms=round(t_u, 4), bitmap_ms=round(t_b, 4),
                         unique_s_per_design=round(t_u * STEPS / 1000.0, 4),
                         bitmap_s_per_design=round(t_b * STEPS / 1000.0, 4),
                         gather_identical=same_g, pos_identical=same_p,
                         u_max_per_tag={t: tag_u[t][Q] for t in all_tags},
                         u_bucket_bound=tile(max(tag_u[t][Q] for t in all_tags))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(tag=TAG, reps=REPS, steps=STEPS, L=L, n_key=NK, k_sparse=K,
                                   threads=1, host=os.uname().nodename, tags=all_tags,
                                   sweep=rows),
                              indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
