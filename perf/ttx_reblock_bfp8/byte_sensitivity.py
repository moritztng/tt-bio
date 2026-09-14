#!/usr/bin/env python3
"""Is the channel move BYTE-bound or TRANSACTION-bound? The control that decides the bfp8 question.

The bfp8 proposal for this path is "a pure DRAM-traffic op, so half the precision is half the bytes
is half the time". That premise is testable without writing a bfp8 kernel at all, and testing it
first is the whole point: the kernel's own comment already claims the binding resource is NOC
transaction count on the writer RISC rather than bytes, and cites a bf16 -> fp32 control for it, but
no such record exists anywhere in the repo -- only the sentence. So it gets re-measured here.

fp32 is the right control and bfp8 is not. Going bf16 -> fp32 DOUBLES every byte this op moves
(2 KB tiles -> 4 KB, DRAM reads and writes alike) and leaves the transaction count EXACTLY where it
was: the writer still gathers 64 face-rows per output tile and still issues one contiguous tile
write, only each transfer is 64 B instead of 32 B. So it separates the two candidate limiters
cleanly, in the direction that needs no new kernel:

    time scales with bytes      -> fp32 costs ~2x, and bfp8 at half the bytes is worth building
    time is flat in bytes       -> the limiter is elsewhere, and bfp8 cannot pay

and the second reading settles it against bfp8 with margin, because real bfp8 makes the binding
resource WORSE, not neutral. A bfp8 tile shares one 8-bit exponent per 16-datum group and a face-row
is exactly one such group, so the gather stays group-aligned and correct -- but every one of those 64
face-row copies becomes two transactions, 16 mantissa bytes to one address plus 1 exponent byte to a
different one. 64 per output tile becomes 128. It also breaks the
`noc_async_read_one_packet_set_state` optimisation this kernel is built around, which writes ONE
transfer length per invocation and can no longer, with two.

Both legs also run `ttnn.permute` on a bfloat8_b tensor at the same shapes. That needs no kernel
change and reads the byte sensitivity of the FALLBACK, so the conclusion does not rest on one
instrument: if the stock op is also flat in bytes here, the path is not bandwidth-limited at all.

PROTOCOL. Every fp32 leg is bracketed by a bf16 control measured immediately before and immediately
after it and is reported against the mean of its own two brackets, because position in the rep is
worth up to 1.70x on this very op (`perf/ttx_splitwork/core_count_sweep.py` measured seven identical
configurations at a 1.70x spread when they were not bracketed). A leg whose two brackets disagree by
more than --bracket-tol is printed as unusable rather than as a number. Warm 3, median of 7, one
`ttnn.synchronize_device` before the clock starts and one before it stops.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:ttx-reblock-bfp8-pairtrack \
        ~/tt-bio/env/bin/python3 perf/ttx_reblock_bfp8/byte_sensitivity.py \
            --out perf/ttx_reblock_bfp8/byte_sensitivity_qb2c3.json
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


def prime():
    """Absorb the one-time cost the FIRST timed shape would otherwise carry.

    Without it the opening row read a 144% bracket spread and was correctly printed unusable: three
    warm iterations cover the per-descriptor JIT but not the process-wide program-cache and
    allocator warmup, which only the first shape in a run ever pays.
    """
    for dt in (ttnn.bfloat16, ttnn.float32):
        for d in ("fwd", "back"):
            _leg(d, 320, 32, dt, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM)
        _stock("fwd", 320, 32, dt, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM)
    _stock("fwd", 320, 32, ttnn.bfloat8_b, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM)


def med(fn):
    """Median of REPS with every intermediate output freed.

    Freeing is not hygiene here: this op allocates its own output, so holding warm + timed results
    live grows allocator occupancy across the run and by a different amount per arm. The last
    result is returned live for the caller's correctness check and freed there.
    """
    ts = []
    o = None
    for i in range(WARM + REPS):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(DEV)
        dt = time.perf_counter() - t0
        if i >= WARM:
            ts.append(dt)
        if i < WARM + REPS - 1:
            ttnn.deallocate(o)
    return statistics.median(ts), o


def mc(buf):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buf)


def _bytes_moved(n, c, elem):
    """One read and one write of the whole tensor, at tile granularity.

    Nt = ceil(N/32) both ways, so a ragged N is priced on the padded tile grid the kernel actually
    touches, not on the logical shape.
    """
    nt = (n + 31) // 32
    ct = c // 32
    return 2 * nt * nt * ct * 32 * 32 * elem


def _leg(direction, n, c, dtype, buf_in, buf_out):
    """One timed call of the custom kernel at `dtype`, with its correctness check.

    Returns (seconds, elem_bytes, exact) where `exact` is the custom kernel against the wheel's own
    permute at the SAME dtype -- the only comparison that means anything, since the whole question
    is what changes when the width changes.
    """
    elem = rp._ELEM_BYTES[dtype]
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    shape = (1, n, n, c) if direction == "fwd" else (1, c, n, n)
    perm = (0, 3, 1, 2) if direction == "fwd" else (0, 2, 3, 1)
    t = torch.randn(*shape, dtype=tdt)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype,
                        memory_config=mc(buf_in))
    out_mc = mc(buf_out)
    prev = rp.set_dtype(dtype)
    try:
        call = (rp.reblock_permute if direction == "fwd" else rp.reblock_permute_back)
        dt, got = med(lambda: call(x, out_mc))
        ref = ttnn.permute(x, perm, memory_config=out_mc)
        exact = torch.equal(ttnn.to_torch(got), ttnn.to_torch(ref))
        ttnn.deallocate(ref)
        ttnn.deallocate(got)
    finally:
        rp.set_dtype(prev)
        ttnn.deallocate(x)
    return dt, elem, exact


def _stock(direction, n, c, dtype, buf_in, buf_out):
    """`ttnn.permute` alone at `dtype` -- the fallback's own byte sensitivity, no kernel involved."""
    tdt = {ttnn.float32: torch.float32}.get(dtype, torch.bfloat16)
    shape = (1, n, n, c) if direction == "fwd" else (1, c, n, n)
    perm = (0, 3, 1, 2) if direction == "fwd" else (0, 2, 3, 1)
    t = torch.randn(*shape, dtype=tdt)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype,
                        memory_config=mc(buf_in))
    out_mc = mc(buf_out)
    dt, o = med(lambda: ttnn.permute(x, perm, memory_config=out_mc))
    ttnn.deallocate(o)
    ttnn.deallocate(x)
    return dt


def run_shape(direction, n, c, buf_in, buf_out, tol, results):
    tag = f"{direction} N={n} C={c} {buf_in.name}->{buf_out.name}"
    print(f"\n--- {tag}", flush=True)
    row = {"direction": direction, "n": n, "c": c,
           "buf_in": buf_in.name, "buf_out": buf_out.name}

    # bracket / leg / bracket, so the fp32 reading is against its own neighbourhood
    b0, e2, ex0 = _leg(direction, n, c, ttnn.bfloat16, buf_in, buf_out)
    f, e4, ex4 = _leg(direction, n, c, ttnn.float32, buf_in, buf_out)
    b1, _, ex1 = _leg(direction, n, c, ttnn.bfloat16, buf_in, buf_out)

    ctrl = (b0 + b1) / 2
    spread = abs(b1 - b0) / ctrl
    row.update(
        bf16_bracket_ms=[b0 * 1e3, b1 * 1e3], bf16_ctrl_ms=ctrl * 1e3, bf16_spread=spread,
        fp32_ms=f * 1e3, fp32_over_bf16=f / ctrl, usable=spread <= tol,
        bf16_exact=bool(ex0 and ex1), fp32_exact=bool(ex4),
        bf16_bytes=_bytes_moved(n, c, 2), fp32_bytes=_bytes_moved(n, c, 4),
        bf16_gbps=_bytes_moved(n, c, 2) / ctrl / 1e9,
        fp32_gbps=_bytes_moved(n, c, 4) / f / 1e9,
    )
    print(f"    custom  bf16 {ctrl*1e3:8.3f} ms ({row['bf16_gbps']:6.1f} GB/s)  "
          f"bracket spread {spread*100:.2f}%  exact={row['bf16_exact']}")
    print(f"    custom  fp32 {f*1e3:8.3f} ms ({row['fp32_gbps']:6.1f} GB/s)  "
          f"2x bytes -> {row['fp32_over_bf16']:.3f}x time  exact={row['fp32_exact']}"
          + ("" if row["usable"] else "   [UNUSABLE: brackets disagree]"))

    # the fallback's own byte sensitivity: bf16 vs bfp8 vs fp32, no custom kernel in the loop
    s0 = _stock(direction, n, c, ttnn.bfloat16, buf_in, buf_out)
    s8 = _stock(direction, n, c, ttnn.bfloat8_b, buf_in, buf_out)
    s4 = _stock(direction, n, c, ttnn.float32, buf_in, buf_out)
    s0b = _stock(direction, n, c, ttnn.bfloat16, buf_in, buf_out)
    sc = (s0 + s0b) / 2
    row.update(stock_bf16_ms=sc * 1e3, stock_bfp8_ms=s8 * 1e3, stock_fp32_ms=s4 * 1e3,
               stock_bf16_spread=abs(s0b - s0) / sc,
               stock_bfp8_over_bf16=s8 / sc, stock_fp32_over_bf16=s4 / sc)
    print(f"    permute bf16 {sc*1e3:8.3f} ms   bfp8 {s8*1e3:8.3f} ms "
          f"({row['stock_bfp8_over_bf16']:.3f}x, 0.53x the bytes)   "
          f"fp32 {s4*1e3:8.3f} ms ({row['stock_fp32_over_bf16']:.3f}x, 2x the bytes)")
    results.append(row)


def main() -> int:
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bracket-tol", type=float, default=0.10)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    DEV = get_device()
    prime()

    g = DEV.compute_with_storage_grid_size()
    out = {
        "host": socket.gethostname(),
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "ttnn": getattr(ttnn, "__version__", "?"),
        "grid": f"{g.x}x{g.y}",
        "bracket_tol": args.bracket_tol,
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": [],
    }

    # The production shapes of this op, both directions. 298 aa and 512 aa are the two rungs the
    # accuracy protocol scores, and C is the trunk's chunk width on an 11x10 grid.
    plan = [
        ("fwd", 298, 32, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM),
        ("fwd", 320, 32, ttnn.BufferType.L1, ttnn.BufferType.L1),
        ("fwd", 512, 32, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM),
        ("fwd", 512, 128, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM),
        ("back", 512, 32, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM),
        ("back", 512, 128, ttnn.BufferType.DRAM, ttnn.BufferType.DRAM),
    ]
    try:
        for direction, n, c, bi, bo in plan:
            try:
                run_shape(direction, n, c, bi, bo, args.bracket_tol, out["rows"])
            except Exception as e:  # a shape the wheel or the gate refuses is data, not a crash
                print(f"    FAILED {direction} N={n} C={c}: {type(e).__name__}: {e}", flush=True)
                out["rows"].append({"direction": direction, "n": n, "c": c,
                                    "error": f"{type(e).__name__}: {e}"})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=1))
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
