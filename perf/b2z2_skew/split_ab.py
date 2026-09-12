#!/usr/bin/env python3
"""Cut the operand chain into k shorter chains and A/B it against the shipped one.

`b2z2-arrival-skew-attack` measured the matmul reader's wait per core per block iteration on WH
and found a hop ramp on top of a floor: the core at hop h waits `intercept + slope * h`, with the
slope worth 0.74 us/hop on the in0 axis. k injectors per axis cut the longest walk from N-1 to
about N/k - 1, so the ramp term shrinks by roughly k, at the price of k DRAM reads of the same
block instead of one -- and the whole DRAM read is 3.2 % of the reader's time with 0.63 % of it
on the wire, so the trade should be strongly favourable if the ramp is on anyone's critical path.

This is NOT the fan-out arm that lost at 0.69918x: that one deleted the chain and put all N
transaction issues on the injector's single RISC. Here every core still issues exactly one
forward, and the only change is where the walk restarts.

Bit-exactness is checked at every k against the k=1 arm's own output: the same bytes reach the
same cores, so `torch.equal` is the bar and a failure is a bug, not a tradeoff.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUT: dict = {"doc": __doc__}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def timed(ttnn, dev, call, reps):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / reps


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "split_ab.json")
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--splits", default="2,4,max")
    ap.add_argument("--reps", type=int, default=20, help="calls per timed slot")
    ap.add_argument("--n-rep", type=int, default=5, help="mirrored A B B A reps")
    a = ap.parse_args()
    OUT_PATH = a.out

    os.environ["TT_BIO_MM_CHAIN_SPLIT"] = "1"
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as MG

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cfg_block = (4, 4, 1, 4, 1)
    ckc = (ttnn.MathFidelity.HiFi4, False, False, False)
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(grid),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3],
                  "m": a.m, "n": a.n, "k": a.k, "block_cfg": list(cfg_block),
                  "reps": a.reps, "n_rep": a.n_rep}
    print(json.dumps(OUT["env"]), flush=True)
    dump()

    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(a.m, a.k).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(a.k, a.n).bfloat16(), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run():
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([a.m, a.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        MG.generic_minimal_matmul(dev, x, w, out, (cfg_block, grid), ckc,
                                  kernel_dir=MG._ARMED_KERNEL_DIR)
        return out

    def arm(k):
        os.environ["TT_BIO_MM_CHAIN_SPLIT"] = str(k)

    arm(1)
    ref_t = run()
    ref = ttnn.to_torch(ref_t)

    # A negative control: the comparison below has to be able to fail. One element of the operand
    # is bumped by 8 eps and the same torch.equal must go false; a round-trip-only rebuild with no
    # bump must stay true, so what the control detects is the bump and not the round trip.
    def rebuilt(bump):
        arm(1)
        xt = ttnn.to_torch(x)
        if bump:
            eps = torch.finfo(torch.bfloat16).eps
            xt.reshape(-1)[0] += (eps * 8 if xt.reshape(-1)[0] == 0 else
                                  xt.reshape(-1)[0].abs() * eps * 8)
        xx = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([a.m, a.n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        MG.generic_minimal_matmul(dev, xx, w, out, (cfg_block, grid), ckc,
                                  kernel_dir=MG._ARMED_KERNEL_DIR)
        got = ttnn.to_torch(out)
        ttnn.deallocate(xx)
        ttnn.deallocate(out)
        return got

    OUT["controls"] = {"negative_control_breaks": not bool(torch.equal(ref, rebuilt(True))),
                       "round_trip_only_holds": bool(torch.equal(ref, rebuilt(False)))}
    print("  controls " + json.dumps(OUT["controls"]), flush=True)
    dump()

    rows = []
    for spec in a.splits.split(","):
        spec = spec.strip()
        arm(spec)
        got_t = run()
        got = ttnn.to_torch(got_t)
        bit_exact = bool(torch.equal(ref, got))
        max_abs = float((ref.float() - got.float()).abs().max())
        built = sorted({(str(e["dims"].get("chain_split")), str(e["dims"].get("chain_walk")))
                        for e in MG._CACHE.values()})

        def slot(k):
            arm(k)
            return timed(ttnn, dev, lambda: ttnn.deallocate(run()), a.reps)

        for _ in range(2):
            slot(1), slot(spec)
        base, armed = [], []
        for _ in range(a.n_rep):
            b1, m1, m2, b2 = slot(1), slot(spec), slot(spec), slot(1)
            base += [b1, b2]
            armed += [m1, m2]
        row = {"split": spec, "bit_exact": bit_exact, "max_abs_diff": max_abs,
               "base_ms": round(st.median(base), 5), "arm_ms": round(st.median(armed), 5),
               "ratio_base_over_arm": round(st.median(base) / st.median(armed), 5),
               "aa_floor": round(st.median(base[0::2]) / st.median(base[1::2]), 5),
               "built": built}
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
        OUT["rows"] = rows
        dump()
        ttnn.deallocate(got_t)

    OUT["verdict"] = {"all_bit_exact": all(r["bit_exact"] for r in rows),
                      "ratio_by_split": {r["split"]: r["ratio_base_over_arm"] for r in rows}}
    print(json.dumps(OUT["verdict"], indent=1), flush=True)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
