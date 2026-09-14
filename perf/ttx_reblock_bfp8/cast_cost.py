#!/usr/bin/env python3
"""What the bfp8 route has to pay before it can save anything: the cast round trip.

`dtype_census.py` established that every tensor reaching the three gates is bf16, in `--fast` as
well as out of it -- 2304 of 2304 and 1152 of 1152 at [1,320,320,32], every one eligible, so the
`*_dtype_layout` clause declines nothing today. That kills the free reading of the bfp8 ask (a
bfp8 caller silently falling back to `ttnn.permute`). What is left is the paid reading: cast the
chunk to bfp8 on the way in and back to bf16 on the way out, so the move itself carries half the
bytes.

That route is only ahead if

    cast_in + bfp8_move + cast_out  <  bf16_move

and both casts read and write the whole tensor exactly like the move does, so the inequality is
close to `2 * move + bfp8_move < move` unless a cast is much cheaper per byte than the move. This
prices both sides at the shapes production actually runs, with the move measured in the same
process so the ratio is not across runs.

Warm 3, median of 7, synchronised both sides of the clock. The casts are timed alone, not fused
into anything, which is the friendliest possible accounting for the bfp8 route.
"""
from __future__ import annotations

import argparse, json, os, socket, statistics, subprocess, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
import tt_bio.reblock_permute as rp                  # noqa: E402

WARM, REPS = 6, 7
DEV = None


def med(fn):
    ts = []
    o = None
    for i in range(WARM + REPS):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(DEV)
        ts.append(time.perf_counter() - t0) if i >= WARM else None
        if i < WARM + REPS - 1:
            ttnn.deallocate(o)
    ttnn.deallocate(o)
    return statistics.median(ts)


def mc(buf):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buf)


def run(n, c, buf, results):
    tag = f"N={n} C={c} {buf.name}"
    t = torch.randn(1, n, n, c, dtype=torch.bfloat16)
    x16 = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                          memory_config=mc(buf))
    x8 = ttnn.typecast(x16, ttnn.bfloat8_b, memory_config=mc(buf))
    m = mc(buf)

    move16 = med(lambda: rp.reblock_permute(x16, m))
    cast_in = med(lambda: ttnn.typecast(x16, ttnn.bfloat8_b, memory_config=m))
    cast_out = med(lambda: ttnn.typecast(x8, ttnn.bfloat16, memory_config=m))
    move16b = med(lambda: rp.reblock_permute(x16, m))   # bracket the casts

    ctrl = (move16 + move16b) / 2
    overhead = (cast_in + cast_out) / ctrl
    row = {"n": n, "c": c, "buf": buf.name,
           "move_bf16_ms": ctrl * 1e3, "move_bracket_ms": [move16 * 1e3, move16b * 1e3],
           "move_spread": abs(move16b - move16) / ctrl,
           "cast_in_ms": cast_in * 1e3, "cast_out_ms": cast_out * 1e3,
           "casts_over_move": overhead}
    print(f"  {tag:28s} move(bf16) {ctrl*1e3:7.3f} ms   "
          f"cast in {cast_in*1e3:7.3f} + out {cast_out*1e3:7.3f} = "
          f"{(cast_in+cast_out)*1e3:7.3f} ms  = {overhead:5.2f}x the move "
          f"(bracket spread {row['move_spread']*100:.2f}%)")
    results.append(row)
    ttnn.deallocate(x8)
    ttnn.deallocate(x16)


def main() -> int:
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    DEV = get_device()
    # Absorb the process-wide one-time cost, which only the first timed shape ever pays and which
    # showed up as a 48.8% bracket spread on the opening rows of the first run.
    _p = ttnn.from_torch(torch.randn(1, 320, 320, 32, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16,
                         memory_config=mc(ttnn.BufferType.DRAM))
    med(lambda: rp.reblock_permute(_p, mc(ttnn.BufferType.DRAM)))
    med(lambda: ttnn.typecast(_p, ttnn.bfloat8_b, memory_config=mc(ttnn.BufferType.DRAM)))
    ttnn.deallocate(_p)
    g = DEV.compute_with_storage_grid_size()
    out = {"host": socket.gethostname(),
           "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": f"{g.x}x{g.y}",
           "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
           "rows": []}
    # [1,320,320,32] is what a 298 aa boltz-2 fold runs, from dtype_census; the 512 rows are the
    # other rung the accuracy protocol scores.
    plan = [(320, 32, ttnn.BufferType.L1),
            (320, 32, ttnn.BufferType.DRAM),
            (512, 32, ttnn.BufferType.DRAM),
            (512, 128, ttnn.BufferType.DRAM)]
    for n, c, buf in plan:
        try:
            run(n, c, buf, out["rows"])
        except Exception as e:
            print(f"  FAILED N={n} C={c} {buf.name}: {type(e).__name__}: {e}")
            out["rows"].append({"n": n, "c": c, "buf": buf.name,
                                "error": f"{type(e).__name__}: {e}"})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
