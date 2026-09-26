#!/usr/bin/env python3
"""Leg 2: grade `tt_bio.rne_add` against float64, not against `ttnn.add`.

The claim is bit-exactness to `round_rne_bf16(exact_sum)` computed in float64 on the host. The
four cases are `perf/bcx_p10_calls/rne_add_probe.py`'s, imported rather than restated, so this
grades the kernel on exactly the pairs that condemned every cheaper ttnn spelling -- including
REAL ties built as `x + halfulp(x)`, because `x + x` is exact in every format and tests nothing.

The kernel has two compile-time arms whose correct settings cannot be read off the LLK with
confidence -- which unit adds, and which unit rounds float32 down to bfloat16 -- so all four
combinations are run and the table decides. `reblock_permute_gated`'s header records the PACKER
breaking ties away from zero on a float32 DEST where ttnn breaks them to even; ROUND_MODE=1
rounds in the SFPU first so the packer never sees a tie. That is the claim this measures.

It also reads every INPUT tensor back after each program and compares its norm to the norm before.
A `generic_op` core placed without runtime arguments reads its output base as address 0 and writes
into the bottom of DRAM, silently, into whatever tensor the allocator put there -- the defect that
cost `bcx-p10-tabwd` four bisects (`state/perf10/bcx-TABWD.md`). An untouched input is the check
that catches it.
"""
import argparse
import json
import pathlib
import sys
import threading
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_calls"))

import ttnn                                                             # noqa: E402
from tt_bio import rne_add                                              # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                     # noqa: E402
from rne_add_probe import bf16, tie_pair, odd_mantissa, N               # noqa: E402

ARMS = [(0, 0), (0, 1), (1, 0), (1, 1)]
ARM_NAME = {
    (0, 0): "FPU add, packer rounds",
    (0, 1): "FPU add, SFPU rounds to even",
    (1, 0): "SFPU add, packer rounds",
    (1, 1): "SFPU add, SFPU rounds to even",
}


class Clock:
    """AICLK sampled DURING the timed section, off the card's own class node."""

    def __init__(self, card):
        self.path = pathlib.Path(
            "/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % card)
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(int(self.path.read_text().strip()))
            except (OSError, ValueError):
                pass
            self._stop.wait(0.05)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)

    def report(self):
        s = self.samples
        if not s:
            return {"n": 0}
        return {"n": len(s), "min": min(s), "max": max(s),
                "median": sorted(s)[len(s) // 2]}


def to_dev(dev, t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cases(n):
    g = torch.Generator().manual_seed(7)
    r1, r2 = bf16(torch.randn(n, generator=g)), bf16(torch.randn(n, generator=g))
    big = bf16(torch.randn(n, generator=g) * 64.0)
    small = bf16(torch.randn(n, generator=g) * 0.25)
    tx, th = tie_pair(n)
    return {
        "random bf16 pairs": (r1, r2),
        "ratio 256x": (big, small),
        "REAL ties (x + halfulp)": (tx, th),
        "REAL ties, negated": (-tx, -th),
    }


def grade(dev, n):
    out = {}
    cs = cases(n)
    for arm in ARMS:
        rne_add.ADD_MODE, rne_add.ROUND_MODE = arm
        rne_add._CACHE.clear()
        totals, per_case = 0, {}
        for name, (x_t, u_t) in cs.items():
            a = to_dev(dev, x_t.reshape(1, 1, N, N))
            b = to_dev(dev, u_t.reshape(1, 1, N, N))
            # `from_torch` must have rounded nothing, or the comparison grades the host.
            assert torch.equal(ttnn.to_torch(a).to(torch.float32), x_t.reshape(1, 1, N, N))
            assert torch.equal(ttnn.to_torch(b).to(torch.float32), u_t.reshape(1, 1, N, N))
            na, nb = float(x_t.double().norm()), float(u_t.double().norm())
            got = ttnn.to_torch(rne_add.rne_add(a, b)).to(torch.float32)
            # The bottom-of-DRAM check: the kernel reads its operands and must not write them.
            na2 = float(ttnn.to_torch(a).to(torch.float64).norm())
            nb2 = float(ttnn.to_torch(b).to(torch.float64).norm())
            ref = (x_t.double() + u_t.double()).to(torch.bfloat16).to(
                torch.float32).reshape(1, 1, N, N)
            d = int((got != ref).sum())
            totals += d
            per_case[name] = {"differ": d, "of": ref.numel(),
                              "pct": 100.0 * d / ref.numel(),
                              "input_norms_moved": [na != na2, nb != nb2]}
            for t in (a, b):
                ttnn.deallocate(t)
        out[str(arm)] = {"name": ARM_NAME[arm], "total_differ": totals, "cases": per_case}
    return out


def bandwidth(dev, shape, reps, arm):
    """Achieved GB/s on the residual's own shape, operand + result bytes."""
    rne_add.ADD_MODE, rne_add.ROUND_MODE = arm
    rne_add._CACHE.clear()
    g = torch.Generator().manual_seed(11)
    a = to_dev(dev, bf16(torch.randn(*shape, generator=g)))
    b = to_dev(dev, bf16(torch.randn(*shape, generator=g)))
    out = ttnn.allocate_tensor_on_device(a.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(3):
        rne_add.rne_add(a, b, out=out)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        rne_add.rne_add(a, b, out=out)
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    elems = 1
    for d in shape:
        elems *= d
    gb = 3.0 * elems * 2 / 1e9
    entry = rne_add._CACHE[next(iter(rne_add._CACHE))]
    return {"shape": list(shape), "reps": reps, "median_s": med,
            "gb_per_call": gb, "achieved_gbps": gb / med,
            "cores": entry["num_cores"], "tiles": entry["num_tiles"],
            "gran": rne_add._gran()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()

    # A lone p300 chip is a CUSTOM topology to tt-metal, so a direct opener has to apply the
    # mesh-graph descriptor itself or `open_device` is a TT_FATAL (commit c48e32670).
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=0)
    clk = Clock(args.card)
    try:
        with clk:
            rne_add.set_enabled(True)
            res = {"grade": grade(dev, N * N)}
            best = None
            for arm in ARMS:
                if res["grade"][str(arm)]["total_differ"] == 0:
                    best = arm
                    break
            res["exact_arm"] = list(best) if best else None
            if best is not None:
                res["bandwidth"] = bandwidth(dev, (1, 288, 288, 128), args.reps, best)
        res["aiclk"] = clk.report()
    finally:
        ttnn.close_device(dev)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "grade.json").write_text(json.dumps(res, indent=2))

    print("AICLK during: %s" % res["aiclk"])
    for arm in ARMS:
        g = res["grade"][str(arm)]
        print("\n== ADD_MODE,ROUND_MODE = %s  (%s) ==" % (arm, g["name"]))
        for name, c in g["cases"].items():
            print("  %-26s differ %6d / %d (%6.3f %%)   inputs moved %s"
                  % (name, c["differ"], c["of"], c["pct"], c["input_norms_moved"]))
        print("  TOTAL %d  %s" % (g["total_differ"],
                                  "EXACT" if g["total_differ"] == 0 else "not exact"))
    if res.get("bandwidth"):
        b = res["bandwidth"]
        print("\nBANDWIDTH %s on %d cores, %d tiles, gran %d: %.3f ms, %.2f GB, %.1f GB/s"
              % (b["shape"], b["cores"], b["tiles"], b["gran"], b["median_s"] * 1e3,
                 b["gb_per_call"], b["achieved_gbps"]))
    print("\nEXACT ARM: %s" % res["exact_arm"])


if __name__ == "__main__":
    main()
