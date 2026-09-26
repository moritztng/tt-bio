#!/usr/bin/env python3
"""bcx-p10-bytes, leg 2: what the fp32 residual add costs in bytes, and the cheaper shapes of it.

`AF2PairBlock._residual` already does what the brief asks for -- it accumulates in float32 and
stores bfloat16 -- and that is exactly why it is expensive. The shipped form is FOUR ttnn ops:

    wide  = typecast(x, f32)        read 1S   write 2S
    other = typecast(update, f32)   read 1S   write 2S
    wide  = add_(wide, other)       read 4S   write 2S
    out   = typecast(wide, bf16)    read 2S   write 1S
                                    ---- 8S read + 7S write = 15S

against a plain `ttnn.add_`'s 3S. S is the bfloat16 tensor. The float32 round trip is the whole
of the 178.7 GB `bcx-p10-shape` attributes to this family, and it buys ONE thing: `ttnn.add` on
bfloat16 breaks ties away from zero while torch and JAX round half to even, and the trunk's
error growth is dominated by that (`af2.py:315`, `scripts/af2_port/residual_add_probe.py`).

So the question is not whether float32 is needed. It is whether the float32 has to be
MATERIALISED to DRAM twice on the way in and once on the way out. Three candidates:

    wide    the shipped four-op form, the control
    add32   `ttnn.add(x, u, dtype=float32)` then one typecast down.  7S, 2.14x fewer bytes
    add16   `ttnn.add(x, u, dtype=bfloat16)` alone.                  3S, 5x fewer bytes

`add16` is in the table because it is the one worth ruling in or out by measurement rather than
by the docstring: if the binary op's own dest accumulation is float32 and its pack rounds half
to even, it is the whole lever in one op. If it rounds away from zero it reproduces the bug the
wide path exists to fix, and the bit check says so immediately.

Graded against a float64 host reference AND against the shipped arm elementwise, because for
this op bit-exactness is free and is the cheapest regression signal there is.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "perf" / "bcx_p10_bytes" / "out"

#: The two shapes `_residual` is called on in a BindCraft 2 round at the 288 device axis: the
#: pair track and the MSA track. 9 adds a block, 6 of them on the pair track.
SHAPES = {"pair": (288, 288, 128), "msa": (2, 288, 256)}


def arms(ttnn):
    """Each arm returns `out`, and owns neither operand: the caller reuses both."""
    def wide(x, u):
        cfg = ttnn.DRAM_MEMORY_CONFIG
        w = ttnn.typecast(x, ttnn.float32, memory_config=cfg)
        o = ttnn.typecast(u, ttnn.float32, memory_config=cfg)
        w = ttnn.add_(w, o)
        ttnn.deallocate(o)
        out = ttnn.typecast(w, ttnn.bfloat16, memory_config=x.memory_config())
        ttnn.deallocate(w)
        return out

    def add32(x, u):
        w = ttnn.add(x, u, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.typecast(w, ttnn.bfloat16, memory_config=x.memory_config())
        ttnn.deallocate(w)
        return out

    def add16(x, u):
        return ttnn.add(x, u, dtype=ttnn.bfloat16, memory_config=x.memory_config())

    return {"wide": wide, "add32": add32, "add16": add16}


def bytes_moved(shape, arm):
    """Operand + result bytes, counted the way `bcx-p10-devmap` counts them."""
    n = 1
    for d in shape:
        n *= d
    S = 2.0 * n
    return {"wide": 15.0, "add32": 7.0, "add16": 3.0}[arm] * S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--min-window", dest="min_window", type=float, default=3.0,
                    help="keep repeating until the arm has been on the card this long. A "
                         "0.87 ms call 40 times is a 35 ms window and the AICLK sampler, "
                         "which ticks once a second, lands NO sample inside it -- a number "
                         "with an empty clock is not a measurement on this box")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="residual.json")
    args = ap.parse_args()

    import ttnn
    from perf.bcx_stack import stack as S
    from tt_bio import tenstorrent as tn
    clock = S.Clock()
    # `tt_bio.get_device`, not `ttnn.open_device`: it takes the lease, pins the card and
    # installs the p300 mesh descriptor an ad-hoc script otherwise TT_FATALs without.
    dev = tn.get_device()
    A = arms(ttnn)
    blob = {"card": args.card, "reps": args.reps, "seed": args.seed, "shapes": {},
            "pci": S.sysfs_node()[1], "host": os.uname().nodename,
            "started_utc": time.strftime("%FT%TZ", time.gmtime()),
            "loadavg_start": os.getloadavg()}
    try:
        for label, shape in SHAPES.items():
            torch.manual_seed(args.seed)
            # Residual operands are a running activation and a block update, which are the same
            # order of magnitude. Equal magnitudes are also where the away-from-zero tie break
            # the wide path exists to fix is densest (`residual_add_probe.py`).
            hx = torch.randn(shape, dtype=torch.float32)
            hu = torch.randn(shape, dtype=torch.float32)
            bx, bu = hx.bfloat16(), hu.bfloat16()
            ref = (bx.double() + bu.double())          # exact: two bf16 sum in float64
            x = ttnn.from_torch(bx, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            u = ttnn.from_torch(bu, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

            rows = {}
            for name, fn in A.items():
                for _ in range(3):                    # jit + program cache, untimed
                    ttnn.deallocate(fn(x, u))
                ttnn.synchronize_device(dev)
                walls, spans, out = [], [], None
                start = time.perf_counter()
                while len(walls) < args.reps or time.perf_counter() - start < args.min_window:
                    if out is not None:
                        ttnn.deallocate(out)
                    t0 = time.perf_counter()
                    out = fn(x, u)
                    ttnn.synchronize_device(dev)
                    t1 = time.perf_counter()
                    walls.append(t1 - t0)
                    # `Clock` stamps its samples with `time.time()`, so a span built from
                    # `perf_counter` matches nothing and every window comes back empty.
                    spans.append(time.time())
                got = ttnn.to_torch(out).float()
                ttnn.deallocate(out)
                err = (got.double() - ref)
                rows[name] = {
                    "s": S.dist(walls), "reps": len(walls),
                    # One window over the whole arm, not one per call: the clock ticks at 1 Hz
                    # and every individual call here is under a millisecond.
                    "aiclk": clock.window([(spans[0], spans[-1])]),
                    "loadavg": os.getloadavg()[0],
                    "GB_per_call": bytes_moved(shape, name) / 1e9,
                    "exact_vs_f64": bool(torch.equal(got.bfloat16().double(),
                                                     ref.bfloat16().double())),
                    "n_wrong": int((got.bfloat16() != ref.bfloat16()).sum()),
                    "max_abs_err": float(err.abs().max()),
                    "rel_l2": float(err.norm() / ref.norm())}
                rows[name]["_t"] = got
                print(json.dumps({"shape": label, "arm": name,
                                  "median_ms": round(rows[name]["s"]["median"] * 1e3, 4),
                                  "reps": rows[name]["reps"],
                                  "GB": round(rows[name]["GB_per_call"], 4),
                                  "n_wrong": rows[name]["n_wrong"],
                                  "aiclk": rows[name]["aiclk"]}), flush=True)

            base = rows["wide"].pop("_t")
            for name in rows:
                t = rows[name].pop("_t", None)
                if t is not None:
                    rows[name]["bit_equal_to_wide"] = bool(torch.equal(t, base))
                    rows[name]["n_differ_from_wide"] = int((t != base).sum())
                else:
                    rows[name]["bit_equal_to_wide"] = True
                    rows[name]["n_differ_from_wide"] = 0
            ttnn.deallocate(x)
            ttnn.deallocate(u)
            blob["shapes"][label] = {"shape": list(shape), "arms": rows}
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    finally:
        clock.stop()
        blob["loadavg_end"] = os.getloadavg()
        blob["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
        print(f"wrote {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
