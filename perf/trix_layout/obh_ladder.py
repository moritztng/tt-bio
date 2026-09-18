#!/usr/bin/env python3
"""The `out_block_h` drain-block lever: is the optimum a fixed BLOCK COUNT or a fixed height?

`fixterm-decompose` handed this row 1.2894x on key A (b=16, M=512, K=128, N=512) from pinning
`out_block_h = 8` where ttnn derives `per_core_M`. Its binding falsifier is that a per-shape pin
is a per-model patch unless the choice is DERIVED FROM THE TILE GRID IN ONE PLACE
(`unified-solution-not-per-model-patches`, STANDING), and that the rule must reproduce the
measured minimum on at least three shapes of the class.

tt-bio already ships the same mechanism on the 1D-mcast path and ships it as a literal:
`_pair_proj_l1_rungs` takes `out_block_h = 5` with `per_core_M` rounded up to a multiple of 5,
measured 1.263x. Two data points, two different heights:

    _pair_proj   per_core_M ~= 25, optimum out_block_h = 5   ->  5 blocks
    key A        per_core_M  = 32, optimum out_block_h = 8   ->  4 blocks

so the CANDIDATE RULE under test is "the divisor of per_core_M nearest per_core_M / B" for a
single B near 4, not a fixed height. This harness measures the whole ladder at several shapes and
reports, per shape, the argmin and what each candidate B would have picked. A rule that misses the
measured minimum on any shape of the class is refuted, and this prints that verdict rather than
fitting after the fact.

Not a fold claim. Interleaved arms, A/A twins on the shipped path, AICLK sampled DURING, and
`torch.equal` against the shipped path on every rung -- `out_block_h` is a drain-schedule
parameter and does not touch `in0_block_w`, so every rung walks the same K blocks and the
contraction accumulates in the same order.
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from clocksample import during  # noqa: E402

import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__, f"wrong tt_bio: {tt_bio.__file__}"
from tt_bio import tenstorrent as tt  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 11

# (name, batch, M, K, N). Key A is fixterm's, reproduced here so this row's instrument is checked
# against a number someone else already took. The rest are the class: tall pair-track projections
# at the c_z the shipped models run, spanning per_core_M so a fixed height and a fixed block count
# make DIFFERENT predictions.
SHAPES = [
    ("keyA",  16, 512, 128, 512),   # fixterm's, per_core_M = 32 on 8x8
    ("keyB",  16, 512, 512, 128),   # fixterm's own falsifier shape: below 1.05x kills it
    ("pair320", 1, 320 * 320, 128, 128),
    ("pair512", 1, 512 * 512, 128, 128),
    ("qkv512",  1, 512 * 512, 128, 384),
]

dev = tt.get_device()
g = dev.compute_with_storage_grid_size()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
torch.manual_seed(0)


def dv(t, mc=DRAM):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=mc)


def timeit(fn, warm=3, reps=5):
    """One bracket around `reps` calls, NOT one per call.

    `state/trix/FIXTERM-HANDOFF.md`: a `sync; call; sync` bracket charges a 0.05142 ms host
    launch/drain floor the chip never pays. Amortising it over `reps` puts it under 1 % of a
    0.05 ms op instead of doubling it.
    """
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e3 / reps


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


OUT = {"host": TAG, "reps": REPS, "grid": [g.x, g.y], "shapes": []}

with during(period=1.5) as clk:
    for name, B, M, K, N in SHAPES:
        mt, nt, kt = (B * M) // 32, N // 32, K // 32
        gx, gy = (8, 8) if (mt % 8 == 0 and nt >= 8) else (g.x, g.y)
        per_m, per_n = -(-mt // gy), -(-nt // gx)
        cc = ttnn.CoreCoord(gx, gy)
        x = dv(torch.randn(B * M, K) * 0.05)
        w = dv(torch.randn(K, N) * 0.05)
        ship = lambda: ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DRAM,
                                   dtype=ttnn.bfloat16)
        ref = ttnn.to_torch(ship())

        def pcfg(obh):
            sh = max((h for h in range(min(4 // max(1, per_n), obh), 0, -1) if obh % h == 0),
                     default=1)
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=cc, in0_block_w=kt,
                out_subblock_h=sh, out_subblock_w=per_n, out_block_h=obh, out_block_w=per_n,
                per_core_M=per_m, per_core_N=per_n, transpose_mcast=False, fused_activation=None)

        rungs, legal = [], []
        for obh in divisors(per_m):
            try:
                pc = pcfg(obh)
                got = ttnn.matmul(x, w, compute_kernel_config=KC, memory_config=DRAM,
                                  dtype=ttnn.bfloat16, program_config=pc)
                eq = torch.equal(ttnn.to_torch(got), ref)
                ttnn.deallocate(got)
                legal.append((obh, pc, eq))
            except Exception as exc:                                       # noqa: BLE001
                rungs.append(dict(obh=obh, refused=repr(exc)[:120]))
        # interleave every legal rung with the shipped path inside each rep
        acc = {obh: [] for obh, _, _ in legal}
        acc["ship"], acc["ship2"] = [], []
        for _ in range(REPS):
            acc["ship"].append(timeit(ship))
            for obh, pc, _ in legal:
                acc[obh].append(timeit(
                    lambda pc=pc: ttnn.matmul(x, w, compute_kernel_config=KC, memory_config=DRAM,
                                              dtype=ttnn.bfloat16, program_config=pc)))
            acc["ship2"].append(timeit(ship))
        med = {k: st.median(v) for k, v in acc.items()}
        floor = max(med["ship"], med["ship2"]) / min(med["ship"], med["ship2"])
        best = min((obh for obh, _, _ in legal), key=lambda o: med[o])
        for obh, _, eq in legal:
            rungs.append(dict(obh=obh, blocks=per_m // obh, ms=round(med[obh], 5),
                              x_vs_ship=round(med["ship"] / med[obh], 4), torch_equal=eq))
        # what each candidate block count B would have picked, scored against the measured argmin
        preds = {}
        for Bc in (3, 4, 5, 6):
            pick = min(divisors(per_m), key=lambda d: abs(d - per_m / Bc))
            preds[Bc] = dict(picked=pick, correct=pick == best,
                             x_vs_ship=round(med["ship"] / med[pick], 4) if pick in med else None)
        row = dict(name=name, b=B, m=M, k=K, n=N, grid=[gx, gy], mt=mt, nt=nt, kt=kt,
                   per_core_M=per_m, per_core_N=per_n, ship_ms=round(med["ship"], 5),
                   aa_floor=round(floor, 4), best_obh=best, best_blocks=per_m // best,
                   best_x_vs_ship=round(med["ship"] / med[best], 4), rungs=rungs, rule=preds)
        OUT["shapes"].append(row)
        print(f"\n{name}: [{B}x{M}x{K}]@[{K}x{N}] grid {gx}x{gy} per_core_M={per_m} "
              f"per_core_N={per_n} | ship {med['ship']:.5f} ms  A/A floor {floor:.4f}x")
        for r in rungs:
            if "refused" in r:
                print(f"    obh {r['obh']:3d}  REFUSED {r['refused'][:70]}")
            else:
                print(f"    obh {r['obh']:3d} ({r['blocks']:3d} blocks) {r['ms']:.5f} ms  "
                      f"{r['x_vs_ship']:6.4f}x  torch.equal={r['torch_equal']}")
        print(f"  ARGMIN out_block_h={best} ({per_m // best} blocks), "
              f"{med['ship'] / med[best]:.4f}x vs shipped")
        for Bc, pr in preds.items():
            print(f"    rule 'nearest divisor to per_core_M/{Bc}' picks {pr['picked']:3d} "
                  f"-> {'HITS' if pr['correct'] else 'MISSES'} the argmin")
        ttnn.deallocate(x)
        ttnn.deallocate(w)

OUT["clock"] = clk.summary()
print("\n" + clk.line(0))
# the rule verdict, stated as the falsifier demands: it must hit on EVERY shape of the class
for Bc in (3, 4, 5, 6):
    hits = [s["rule"][Bc]["correct"] for s in OUT["shapes"]]
    OUT.setdefault("rule_verdict", {})[Bc] = dict(hits=sum(hits), of=len(hits), all=all(hits))
    print(f"rule per_core_M/{Bc}: hits {sum(hits)}/{len(hits)} -- "
          f"{'SURVIVES' if all(hits) else 'REFUTED'}")
bad = [r for s in OUT["shapes"] for r in s["rungs"] if r.get("torch_equal") is False]
print("PARITY:", "every rung torch.equal" if not bad else f"BROKEN at {bad}")
p = HERE / f"obh_ladder_{TAG}.json"
p.write_text(json.dumps(OUT, indent=2))
print("WROTE", p)
