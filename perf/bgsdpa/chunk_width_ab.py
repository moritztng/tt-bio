#!/usr/bin/env python3
"""Does the in-projection's channel-chunk width move the trimul's numbers?

`_TRIMUL_MIN_CHUNK` stops `_trimul_inproj_chunk_cap` narrowing the channel chunk below one tile,
so above 2048 padded tokens a trimul that used to run two 16-wide channel passes now runs one
32-wide one. Everything in the module says that is bit-exact -- the chunk is a partition of an
independent-channel sum -- but that argument has only ever been checked across `group` widths
(`perf/trimul_root/group_ab.py`), never across `chunk` widths on the DRAM path.

This runs the real module on real (random) weights at a size where EVERY width still returns, so
the widths can be compared at all: above 2048 the narrow arm wedges the device and there is no
"before" to diff against. `torch.equal`, not a PCC.

    TT_VISIBLE_DEVICES=2 python3 perf/bgsdpa/chunk_width_ab.py --out perf/bgsdpa/chunk_ab.json
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

import torch

import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

ROOT = pathlib.Path(__file__).resolve().parents[2]

# `build()` already assembles a TriangleMultiplication on random weights from (cz, hidden).
_spec = importlib.util.spec_from_file_location(
    "inproj_shape_read", ROOT / "perf" / "trimul_kernel" / "inproj_shape_read.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["inproj_shape_read"] = _mod
_spec.loader.exec_module(_mod)
build = _mod.build

# (cz, hidden) of the models that reach this path, and the byte caps that pick a width for them.
# 1 GiB is what ships; the smaller caps are how a narrower chunk is reached at a size that still
# returns, which is what makes the comparison possible.
SHAPES = [("boltzgen", 64, 32), ("opendde", 384, 384)]
CAPS = [1024, 200, 100, 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    shipped = T._TRIMUL_INPROJ_FUSED_BYTES
    rows = []
    for name, cz, hidden in SHAPES:
        for ending in (False, True):
            tm = build(cz, hidden, ckc)
            tm.ending = ending
            torch.manual_seed(0)
            z = ttnn.from_torch(torch.randn(1, args.n, args.n, cz), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16)
            ref = None
            for cap_mib in CAPS:
                T._TRIMUL_INPROJ_FUSED_CAP.clear()
                T._TRIMUL_INPROJ_FUSED_BYTES = cap_mib * 2 ** 20
                c = T._trimul_inproj_chunk_cap(
                    args.n, hidden, 1, T._trimul_chunk_size(args.n, hidden, 1))
                g = T._trimul_inproj_group(args.n, hidden, 1, hidden // c)
                t0 = time.perf_counter()
                out = tm(z, None)
                ttnn.synchronize_device(dev)
                ms = round((time.perf_counter() - t0) * 1e3, 3)
                h = ttnn.to_torch(out)
                ttnn.deallocate(out)
                if ref is None:
                    ref, eq, maxabs = h, True, 0.0
                else:
                    eq = bool(torch.equal(h, ref))
                    maxabs = float((h.float() - ref.float()).abs().max())
                row = dict(model=name, cz=cz, hidden=hidden, n=args.n, ending=ending,
                           cap_mib=cap_mib, chunk=c, group=g, slice_c=c * g,
                           tile_aligned=(c * g) % 32 == 0, ms=ms,
                           bit_exact_vs_shipped_cap=eq, max_abs=maxabs)
                rows.append(row)
                print(json.dumps(row), flush=True)
            ttnn.deallocate(z)
            del tm
    T._TRIMUL_INPROJ_FUSED_BYTES = shipped
    T._TRIMUL_INPROJ_FUSED_CAP.clear()
    bad = [r for r in rows if not r["bit_exact_vs_shipped_cap"]]
    with open(args.out, "w") as fh:
        json.dump({"n": args.n, "rows": rows, "all_bit_exact": not bad,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=2)
    print(f"ALL_BIT_EXACT {not bad}  arms {len(rows)}")


main()
