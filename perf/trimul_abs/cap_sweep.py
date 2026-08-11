#!/usr/bin/env python3
"""E1's gate: how wide may the fused in-projection be, as a function of the shape?

The banked role-major `_TRIMUL_INPROJ_GROUP=8` is 1.077x on a 512 aa fold and shipped OFF for one
reason: its fused projection output is 512 MiB at 512 aa against 64 MiB at G=1, and the large abag
targets (9i3p at 973 aa, 9j4c at 1136 aa) already sit near a refusal. This measures the thing
directly instead of reasoning about it: for each (N, group) it runs the real
`TriangleMultiplication` on real layer-0 weights and reports whether it allocated at all, the DRAM
high-water mark inside the call, and the smallest per-bank contiguous free block left.

Two details that make the number mean something:

* the peak is sampled INSIDE the channel loop (around `ttnn.chunk`, which is the moment the
  537 MB fused projection and its four pieces are all live), not at the module boundary. The
  module's own `dram_peak` tags are outside the loop and cannot see it.
* `--ballast GiB` allocates and holds that much DRAM before the sweep. A standalone trimul sees a
  pristine 31.9 GiB pool; a real 9j4c fold has ~5.9 GiB live and a fragmented one
  (state/capacity_9j4c_dram2.log). Fragmentation, not total free, is what refuses an interleaved
  allocation, so the sweep is run with ballast to look like the fold it must not break.
"""
import argparse, json, time
from pathlib import Path
import sys

import torch
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="512,576,640,704,768,973,1136")
ap.add_argument("--groups", default="1,2,4,8")
ap.add_argument("--ballast", type=float, default=0.0, help="GiB of DRAM held during the sweep")
ap.add_argument("--time", type=int, default=704, help="also time the call for N <= this")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
tm = layer.triangle_multiplication_start
torch.manual_seed(0)


def mem():
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    used = (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks
    lcf = mv.largest_contiguous_bytes_free_per_bank
    if isinstance(lcf, (list, tuple)):
        lcf = min(lcf)
    return used, lcf, mv.total_bytes_per_bank * mv.num_banks


_, _, TOTAL = mem()

ballast = []
if a.ballast:
    # one 1 GiB tensor at a time, so the hold is fragmented the way a fold's is rather than one
    # contiguous slab
    per = 2 ** 30 // 2 // (1024 * 1024)          # bf16 elements per GiB, as 1024x1024 tiles
    for _ in range(int(a.ballast)):
        ballast.append(ttnn.from_torch(torch.zeros(1, per, 1024, 1024, dtype=torch.bfloat16),
                                       layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16))

BASE_USED, BASE_LCF, _ = mem()
print(f"[base] used={BASE_USED/2**30:.3f} GiB of {TOTAL/2**30:.1f}, "
      f"maxfree={BASE_LCF/2**20:.0f} MiB/bank", flush=True)

PEAK = [0, 1 << 62]
SAMPLE = [False]
_chunk = ttnn.chunk


def chunk(*args, **kw):
    out = _chunk(*args, **kw)
    if SAMPLE[0]:
        used, lcf, _ = mem()
        PEAK[0] = max(PEAK[0], used)
        PEAK[1] = min(PEAK[1], lcf)
    return out


ttnn.chunk = chunk

OUT = {"c_z": c_z, "hidden": tm._hidden, "grid": list(COMPUTE_GRID_MAIN),
       "total_dram": TOTAL, "ballast_gib": a.ballast, "base_used": BASE_USED,
       "base_maxfree": BASE_LCF, "rows": []}

for N in [int(s) for s in a.sizes.split(",")]:
    Ht = (N + 31) // 32 * 32
    try:
        z = ttnn.from_torch(torch.randn(1, Ht, Ht, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
    except Exception as e:                                              # noqa: BLE001
        OUT["rows"].append({"n": N, "group": None, "ok": False, "err": f"z: {e}"[:200]})
        print(f"N={N}: z refused", flush=True)
        continue
    for g in [int(s) for s in a.groups.split(",")]:
        T._TRIMUL_INPROJ_GROUP = g
        # the gate under test must not interfere with the sweep: measure the raw group
        T._TRIMUL_INPROJ_FUSED_BYTES = 1 << 62
        row = {"n": N, "nt": Ht, "group": g}
        PEAK[0], PEAK[1] = 0, 1 << 62
        SAMPLE[0] = True
        try:
            out = tm(z, None)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(out)
            row["ok"] = True
        except Exception as e:                                          # noqa: BLE001
            row["ok"] = False
            row["err"] = str(e)[:300]
        SAMPLE[0] = False
        row["peak"] = PEAK[0]
        row["maxfree"] = None if PEAK[1] == 1 << 62 else PEAK[1]
        row["fused_bytes"] = 4 * g * T.TRIANGLE_MULT_CHUNK_SIZE * Ht * Ht * 2
        if row["ok"] and N <= a.time:
            for _ in range(2):
                ttnn.deallocate(tm(z, None))
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = [tm(z, None) for _ in range(4)]
            ttnn.synchronize_device(dev)
            row["ms"] = (time.perf_counter() - t0) * 1e3 / 4
            for o in outs:
                ttnn.deallocate(o)
        print(f"N={N} G={g}: ok={row['ok']} peak={row['peak']/2**30:.3f} GiB "
              f"maxfree={(row['maxfree'] or 0)/2**20:.0f} MiB fused={row['fused_bytes']/2**20:.0f} MiB "
              f"{('%.3f ms' % row['ms']) if 'ms' in row else ''}"
              f"{(' ERR ' + row.get('err', '')[:90]) if not row['ok'] else ''}", flush=True)
        OUT["rows"].append(row)
        a.out.write_text(json.dumps(OUT, indent=1))
    ttnn.deallocate(z)

a.out.write_text(json.dumps(OUT, indent=1))
print("wrote", a.out)
