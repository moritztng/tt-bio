#!/usr/bin/env python3
"""Rank the census's batched-matmul classes by MEASURED device time, and place each on a roof.

Input is a census JSON from mm_census.py. For every class that trips the batched signature this
replays the exact padded shape / dtype / buffer type on synthetic DRAM-interleaved operands and
times three arms with a device synchronise on both sides:

    auto     -- exactly as the model calls it today (no core_grid, no program_config)
    grid     -- + core_grid=CORE_GRID_MAIN. In ttnn this is not a hint: `generate_matmul_program_config`
                routes any call with a user grid into `create_matmul_program_config`, whose
                batched-B branch returns MatmulMultiCoreReuseProgramConfig{per_core_M=Mt,
                per_core_N=Nt, in0_block_w=1} -- the one factory that splits B across cores.
    reuse    -- the same factory with D3's tuned in0_block_w (2 when Kt is even, else 1), skipped
                when G1's correctness predicate rejects it.

G1's predicate, and it is a correctness gate not a perf one: both dataflow kernels advance a whole
BATCH stride per per-core loop iteration while the factory passes a per-core BLOCK count, so the
result is wrong unless `per_core_M == Mt` (one block is one batch element) or the total block count
is <= the core count (each core gets exactly one block, so the bad increment never runs).

Per-call microseconds are multiplied by the census call count to give ms per 298 aa fold, which is
the scoreboard's currency. An op-isolated replay is a COST attribution, not a promised gain: D3
measured the same op-level ratio realise 101% at the wall on opendde and 14% on protenix-v2.

    TT_VISIBLE_DEVICES=0 python3 perf/attn_sites/mm_replay.py \
        --census perf/attn_sites/census_protenix-v2_298.json \
        --roofs  perf/attn_sites/roofs_pc0.json \
        --out    perf/attn_sites/rank_protenix-v2_298.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

TILE = 32
DT = {"BFLOAT16": (ttnn.bfloat16, 2), "FLOAT32": (ttnn.float32, 4),
      "BFLOAT8_B": (ttnn.bfloat8_b, 1), "UINT32": (ttnn.uint32, 4)}


def timed(fn, dev, warm=3, reps=20):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del outs
    return dt


def subblock(m, n, fp32_dest_acc):
    """Largest legal (h, w) with h*w <= DEST tiles (4 with fp32_dest_acc, else 8)."""
    cap = 4 if fp32_dest_acc else 8
    best = (1, 1)
    for h in range(1, m + 1):
        for w in range(1, n + 1):
            if m % h == 0 and n % w == 0 and h * w <= cap and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def reuse_config(mt, kt, nt, batch, cores, fp32_dest_acc):
    """MatmulMultiCoreReuseProgramConfig, or None when no legal per_core_M is safe.

    per_core_N is not a knob: matmul_device_operation.cpp asserts N == per_core_N for this factory.
    per_core_M sweeps the divisors of Mt, largest first, and each candidate must clear G1's
    correctness predicate before it is allowed to compete on speed.
    """
    in0_bw = 2 if kt % 2 == 0 else 1
    for pcm in sorted((d for d in range(1, mt + 1) if mt % d == 0), reverse=True):
        blocks = batch * (mt // pcm)
        if not (pcm == mt or blocks <= cores):
            continue
        h, w = subblock(pcm, nt, fp32_dest_acc)
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(CORE_GRID_MAIN.x, CORE_GRID_MAIN.y),
            in0_block_w=in0_bw, out_subblock_h=h, out_subblock_w=w,
            per_core_M=pcm, per_core_N=nt), pcm, blocks, in0_bw
    return None, None, None, None


def roofline(row, roofs):
    b, mt, kt, nt = row["batch"], row["mt"], row["kt"], row["nt"]
    ea = DT[row["a"]["dtype"]][1]
    eb = DT[row["b"]["dtype"]][1]
    eo = DT[(row["out"] or row["a"])["dtype"]][1]
    flops = 2.0 * b * mt * kt * nt * TILE ** 3
    ba = b * mt * kt * TILE ** 2 * ea
    bb = b * kt * nt * TILE ** 2 * eb
    bo = b * mt * nt * TILE ** 2 * eo
    ai = flops / (ba + bb + bo)
    rd, wr = roofs["dram_read_GBs"] * 1e9, roofs["dram_write_GBs"] * 1e9
    comp = roofs["compute_bf16_TFLOPs"] * 1e12 if ea == 2 else roofs["compute_fp32_TFLOPs"] * 1e12
    # G1's kernel-derived overlap model for the reuse factory: in0 is read by one dataflow RISC;
    # in1 reads and output writes are both issued by the other, in one loop with a write barrier
    # per subblock. So the memory floor is max(in0, in1 + out), not the sum of all three.
    t_mem = max(ba / rd, bb / rd + bo / wr)
    t_comp = flops / comp
    return {"flops": flops, "bytes": ba + bb + bo, "ai": ai,
            "balance": comp / rd, "t_mem_s": t_mem, "t_comp_s": t_comp,
            "binding": "compute" if t_comp > t_mem else "memory",
            "floor_s": max(t_comp, t_mem)}


def build(desc, dev):
    dt, _ = DT[desc["dtype"]]
    shp = desc["padded"]
    t = torch.randn(*shp) if dt is not ttnn.uint32 else torch.zeros(*shp, dtype=torch.int32)
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="replay hinted classes too, as controls")
    args = ap.parse_args()

    census = json.loads(Path(args.census).read_text())
    roofs = json.loads(Path(args.roofs).read_text())
    dev = get_device()
    cores = int(dev.compute_with_storage_grid_size().x) * int(dev.compute_with_storage_grid_size().y)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = []
    for row in census["rows"]:
        if not row["trips"] and not args.all:
            continue
        a = build(row["a"], dev)
        b = build(row["b"], dev)
        rec = dict(row)
        rec["roofline"] = roofline(row, roofs)
        try:
            rec["auto_us"] = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=ckc),
                                   dev, reps=args.reps) * 1e6
        except Exception as e:                                            # noqa: BLE001
            rec["auto_us"], rec["auto_err"] = None, f"{type(e).__name__}: {e}"[:200]
        try:
            rec["grid_us"] = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                       core_grid=CORE_GRID_MAIN),
                                   dev, reps=args.reps) * 1e6
        except Exception as e:                                            # noqa: BLE001
            rec["grid_us"], rec["grid_err"] = None, f"{type(e).__name__}: {e}"[:200]
        pc, pcm, blocks, bw = reuse_config(row["mt"], row["kt"], row["nt"], row["batch"], cores, True)
        # A rejected program config raises TT_FATAL and aborts the process, so it is NOT wrapped in
        # try/except -- the predicate above has to be right before the call is made (G1).
        if pc is not None:
            rec.update(reuse_per_core_M=pcm, reuse_blocks=blocks, reuse_in0_block_w=bw)
            rec["reuse_us"] = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                        program_config=pc),
                                    dev, reps=args.reps) * 1e6
        else:
            rec["reuse_us"] = None
            rec["reuse_skip"] = "no per_core_M clears G1's correctness predicate"
        for arm in ("auto", "grid", "reuse"):
            us = rec.get(f"{arm}_us")
            rec[f"{arm}_ms_fold"] = (us * row["n"] / 1000.0) if us else None
            rec[f"{arm}_pct_floor"] = (rec["roofline"]["floor_s"] * 1e6 / us * 100.0) if us else None
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        out.append(rec)
        print(f"{row['site']:<46} n={row['n']:<6} auto={rec['auto_us']} us "
              f"grid={rec['grid_us']} reuse={rec['reuse_us']}", flush=True)

    out.sort(key=lambda r: -(r["auto_ms_fold"] or 0))
    Path(args.out).write_text(json.dumps({"grid_cores": cores, "roofs": roofs, "rows": out}, indent=1))
    tot = sum(r["auto_ms_fold"] or 0 for r in out)
    print(f"\ntotal replayed device time in the tripping classes: {tot:.1f} ms/fold", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
