#!/usr/bin/env python3
"""Does `_BATCHED_MATMUL_SATURATION_BLOCKS = 32` need a grid term?

The constant is an absolute output-block count fitted on qb1's 130-core p150a and read by
`_batched_matmul_search`, which already holds `cores = gx * gy` two lines above it and uses the
core count in its legality filter but not in its saturation target. This script asks the only
question that decides whether that is a defect: hold the silicon fixed, move the core count, and
see whether the block count that wins moves with it.

`TT_BIO_FORCE_GRID` does the moving, so 13x10 / 11x10 / 8x9 are three grids on ONE Blackhole
p150a. An 8x9 reading here is NOT a Wormhole reading -- same DRAM, same banks, same clock, same
kernels, only the grid differs. It bounds the grid term and nothing else.

Arms are the distinct `per_core_M` values the constant can select, entered in a fixed order, whole
set once per block, with the shipped pick entered twice under two names as the A/A floor. Every arm
is `torch.equal`-checked against the shipped one: `per_core_M` partitions output rows and
`in0_block_w` does not move, so bit-exactness is expected and its absence would mean this is not
the lever it looks like.

    python3 perf/roof_wrong_part/sat_blocks.py --out sat_13x10.json
    TT_BIO_FORCE_GRID=8,9 python3 perf/roof_wrong_part/sat_blocks.py --out sat_8x9.json
"""
import argparse
import json
import os
import statistics
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG
# Every SAT value that can select a different per_core_M at any shape in the ladder.
SAT_CANDIDATES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024)


def cases(ladder):
    """The batched-matmul classes a Boltz-2 fold issues, at each rung of the ladder.

    Trunk `AttentionPairBias` is c_s=384 / 16 heads, so head_dim 24 padded to 32; those two
    matmuls are 264 calls and 0.227 s above roof at 512 aa and are the only Boltz-2 site with more
    than one legal `per_core_M`. The token-DiT pair (c_s=768, head_dim 48 padded to 64) is dark on
    Boltz-2 today because `BOLTZ2_TOKEN_DIT_SDPA` is on, and is carried anyway: it is live on every
    model that does not take that branch, and it is the class the constant was fitted on.
    The atom site is [.., 32, ..] so m_tiles is 1 and no choice exists; it is not swept.
    """
    out = []
    for s in ladder:
        out.append(dict(label=f"trunk APB q@kT {s}aa", live=True, s=s,
                        a=(1, 16, s, 32), b=(1, 16, 32, s)))
        out.append(dict(label=f"trunk APB attn@v {s}aa", live=True, s=s,
                        a=(1, 16, s, s), b=(1, 16, s, 32)))
        out.append(dict(label=f"token DiT q@kT {s}aa", live=False, s=s,
                        a=(1, 16, s, 64), b=(1, 16, 64, s)))
        out.append(dict(label=f"token DiT attn@v {s}aa", live=False, s=s,
                        a=(1, 16, s, s), b=(1, 16, s, 64)))
    return out


def configs_for(batch, mt, kt, nt):
    """{per_core_M: (config, blocks, [SAT values that select it])}, from the engine's own chooser."""
    found = {}
    shipped = T._BATCHED_MATMUL_SATURATION_BLOCKS
    try:
        for sat in SAT_CANDIDATES:
            T._BATCHED_MATMUL_SATURATION_BLOCKS = sat
            T._batched_matmul_search.cache_clear()
            cfg = T._batched_matmul_config(batch, mt, kt, nt, 2)
            if cfg is None:
                continue
            p = int(cfg.per_core_M)
            found.setdefault(p, [cfg, batch * mt // p, []])[2].append(sat)
    finally:
        T._BATCHED_MATMUL_SATURATION_BLOCKS = shipped
        T._batched_matmul_search.cache_clear()
    return found


def timed(fn, iters):
    ttnn.synchronize_device(T.get_device())
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(T.get_device())
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--ladder", default="320,512,768,1024")
    ap.add_argument("--target-ms", type=float, default=12.0,
                    help="per-arm wall inside one block; iters adapt to it")
    a = ap.parse_args()

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    ladder = [int(v) for v in a.ladder.split(",")]
    res = {"grid": list(grid), "cores": cores, "arch": str(dev.arch()),
           "shipped_sat": T._BATCHED_MATMUL_SATURATION_BLOCKS,
           "l1_bank_bytes": int(T._matmul_cb_budget()), "cases": []}

    for case in cases(ladder):
        sa, sb = case["a"], case["b"]
        batch = sa[0] * sa[1]
        mt, kt, nt = -(-sa[2] // 32), -(-sa[3] // 32), -(-sb[3] // 32)
        found = configs_for(batch, mt, kt, nt)
        row = dict(case, batch=batch, Mt=mt, Kt=kt, Nt=nt,
                   in0_block_w=int(T._batched_matmul_block_w(mt, kt, nt)),
                   legal={}, arms={})
        if not found:
            row["note"] = "chooser declines at every SAT; the plain ttnn call is what ships"
            res["cases"].append(row)
            print(f"  {case['label']:28s} declines")
            continue
        shipped_p = int(T._batched_matmul_config(batch, mt, kt, nt, 2).per_core_M)
        row["shipped_per_core_M"] = shipped_p
        row["shipped_blocks"] = batch * mt // shipped_p
        for p, (_c, blocks, sats) in sorted(found.items()):
            row["legal"][str(p)] = {"blocks": blocks, "selected_by_sat": sats}

        A = ttnn.from_torch(torch.randn(*sa, dtype=torch.float32) * 0.1, device=dev,
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(*sb, dtype=torch.float32) * 0.1, device=dev,
                            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)

        arms = [(f"pM={p}", found[p][0]) for p in sorted(found)]
        arms.append((f"pM={shipped_p}_AA", found[shipped_p][0]))   # own-session A/A floor
        arms.append(("ttnn_plain", None))

        def run(cfg):
            return ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg)

        ref = ttnn.to_torch(run(found[shipped_p][0]))
        for name, cfg in arms:
            row["arms"][name] = {"bit_exact": bool(torch.equal(ttnn.to_torch(run(cfg)), ref))}

        one = timed(lambda: run(found[shipped_p][0]), 3)
        iters = max(20, min(3000, int(a.target_ms / max(one, 1e-4))))
        walls = {name: [] for name, _ in arms}
        for _ in range(a.blocks):
            for name, cfg in arms:                                  # fixed arm order
                walls[name].append(timed(lambda c=cfg: run(c), iters))
        for name, _ in arms:
            row["arms"][name].update(ms=round(min(walls[name]), 6),
                                     median_ms=round(statistics.median(walls[name]), 6),
                                     iters=iters)
        for p in sorted(found):
            row["arms"][f"pM={p}"]["blocks"] = found[p][1]
        best = min((v["ms"], k) for k, v in row["arms"].items()
                   if k.startswith("pM=") and not k.endswith("_AA"))
        row["best_arm"] = best[1]
        row["best_blocks"] = row["arms"][best[1]].get("blocks")
        row["x_vs_shipped"] = round(row["arms"][f"pM={shipped_p}"]["ms"] / best[0], 4)
        row["aa_floor"] = round(abs(row["arms"][f"pM={shipped_p}_AA"]["ms"]
                                    / row["arms"][f"pM={shipped_p}"]["ms"] - 1), 5)
        ttnn.deallocate(A)
        ttnn.deallocate(B)
        res["cases"].append(row)
        print(f"  {case['label']:28s} shipped pM={shipped_p} ({row['shipped_blocks']} blk) "
              f"best {best[1]} ({row['best_blocks']} blk) {row['x_vs_shipped']}x  "
              f"A/A {row['aa_floor']*100:.3f}%")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"grid {grid} cores {cores} -> {a.out}")


if __name__ == "__main__":
    main()
