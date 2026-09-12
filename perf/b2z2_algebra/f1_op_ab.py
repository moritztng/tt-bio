#!/usr/bin/env python3
"""The fused trimul tail timed ALONE, direct pack against the staged one.

The module-level A/B cannot see this change: the deleted work is two L1 round trips inside ONE op
of a 24-op chain, and whglx's A/A floor with eight rows on it is 4-5 %. So time the op by itself,
where the same delta is the whole measurement, and interleave the three labels so drift cancels.

`torch.equal` on the op's own output is the parity claim; the module-level run in `f1_direct_ab.py`
is the one that shows the whole TriangleMultiplication is unchanged.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_algebra"))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.trimul_tail as F1                                               # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402
from ledger import CZ                                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--host", default="whglx")
    ap.add_argument("--card", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    ckc = MG.ckc_args(ck)
    grid = tuple(T.COMPUTE_GRID_MAIN)

    g = torch.Generator().manual_seed(5)
    mk = lambda *s: ttnn.from_torch(torch.randn(*s, generator=g), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xa, xb = mk(1, a.n, a.n, a.cz), mk(1, a.n, a.n, a.cz)
    wa, wb = mk(a.cz, a.cz), mk(a.cz, a.cz)

    def call(on):
        F1.DIRECT_PACK = 1 if on else 0
        return F1.fused_tail(xa, xb, wa, wb, ckc, grid)

    res = {"n": a.n, "cz": a.cz, "host": a.host, "card": a.card, "reps": a.reps,
           "arch": "WH" if a.host == "whglx" else "BH", "grid": list(grid),
           "eligible": F1.eligible(xa, xb, wa, wb),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if res["eligible"] is not None:
        print("DECLINED:", res["eligible"])
        Path(a.out).write_text(json.dumps(res, indent=1))
        return

    got = {}
    for on in (False, True):
        o = call(on)
        ttnn.synchronize_device(dev)
        got[on] = ttnn.to_torch(o)
        ttnn.deallocate(o)
    res["parity"] = {"equal": bool(torch.equal(got[False], got[True])),
                     "max_abs": float((got[False].float() - got[True].float()).abs().max())}
    print("parity:", json.dumps(res["parity"]), flush=True)

    labels = [("off", False), ("on", True), ("off2", False)]
    for _ in range(3):                                   # warm both descriptors
        for _, on in labels:
            ttnn.deallocate(call(on))
    ttnn.synchronize_device(dev)

    times = {k: [] for k, _ in labels}
    for _ in range(a.reps):
        for lbl, on in labels:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = call(on)
            ttnn.synchronize_device(dev)
            times[lbl].append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
    med = {k: round(statistics.median(v), 4) for k, v in times.items()}
    # Baseline is the MEAN of the two shipped labels, so a monotone within-rep drift -- which is
    # what a shared box produces -- cannot be read as the lever.
    base = (med["off"] + med["off2"]) / 2
    res["time_ms"] = med
    res["time_raw"] = times
    res["ratio"] = round(base / med["on"], 5)
    res["aa_floor"] = round(med["off"] / med["off2"], 5)
    print("median ms:", json.dumps(med))
    print("ratio (mean shipped)/direct:", res["ratio"], " A/A floor:", res["aa_floor"])
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
