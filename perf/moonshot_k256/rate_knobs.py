#!/usr/bin/env python3
"""What actually sets the achieved rate at K=256 -- the knobs the ceiling model held fixed.

`ceiling-298aa.md` measured every trunk shape at one operating point and concluded the rate is
capped by contraction depth. That point was `MathFidelity.HiFi4, fp32_dest_acc_en=True` for
`ttnn.matmul`, and -- because the sweep passed no `compute_kernel_config` -- whatever
`minimal_matmul` defaults to for the classes minimal_matmul won. Production passes HiFi4 to
minimal_matmul at every site. So the ceiling's denominator and production may not be the same
operating point, and neither has ever been swept.

Four questions, in order of how much they would move the ceiling:
  A. what is this card's compute roof at each fidelity, and is HiFi4 really 4x LoFi here?
  B. what does minimal_matmul default to, and does production's explicit HiFi4 cost time?
  C. does fp32_dest_acc_en=False (dest 4 -> 8 tiles, so wider out_subblocks) raise the rate?
  D. does widening N at fixed K=256 raise the rate -- i.e. is the fusion of parallel
     same-input projections a rate lever the ceiling never priced?

Every arm is run twice, interleaved, so the A/A spread is reported next to every effect.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:moonshot-4x-k256-kernel-rate \
        python3 perf/moonshot_k256/rate_knobs.py out.json
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul


def timed(dev, fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out)


dev = get_device()
g = dev.compute_with_storage_grid_size()
CORES = g.x * g.y
print(f"grid {g.x}x{g.y} = {CORES} cores", flush=True)
res = {"grid": [g.x, g.y], "cores": CORES}
torch.manual_seed(0)

FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2,
       "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}


def ckc(fid, f32=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=FID[fid], fp32_dest_acc_en=f32, packer_l1_acc=True)


# ---------------------------------------------------------------- A: the roof, per fidelity
print("\n== A. square 4096 compute roof by fidelity (bf16 operands, result L1) ==", flush=True)
a = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
b = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
gf = 2 * 4096 ** 3 / 1e9
res["roof_by_fidelity"] = {}
for f32 in (True, False):
    for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi"):
        k = ckc(fid, f32)
        try:
            r = [gf / (timed(dev, lambda: ttnn.deallocate(
                ttnn.matmul(a, b, compute_kernel_config=k, memory_config=L1))) / 1e3) / 1e3
                 for _ in range(2)]
        except Exception as e:
            print(f"  {fid:6s} f32acc={f32}  ERR {str(e)[:70]}", flush=True)
            continue
        key = f"{fid}_f32acc{int(f32)}"
        res["roof_by_fidelity"][key] = {"tflops": [round(x, 2) for x in r]}
        print(f"  {fid:6s} f32acc={int(f32)}  {r[0]:7.2f} / {r[1]:7.2f} TFLOP/s  "
              f"(A/A {abs(r[0]-r[1])/max(r):.2%})", flush=True)
ttnn.deallocate(a); ttnn.deallocate(b)

# ---------------------------------------------- B/C: the real trunk shapes, knob by knob
# The four classes that carry the most arithmetic, at the padded shape they really execute.
SHAPES = [
    ("trimul.in_proj  @[256,128]", (1, 320, 320, 256), (256, 128)),
    ("trimul.out_proj @[256,256]", (1, 320, 320, 256), (256, 256)),
    ("triatt.qkv      @[256,768]", (320, 320, 256), (256, 768)),
    ("transition.up   @[256,1024]", (1, 30, 320, 256), (256, 1024)),
]
print("\n== B/C. trunk shapes: minimal_matmul, fidelity x fp32_dest_acc ==", flush=True)
res["trunk_knobs"] = {}
for label, ash, bsh in SHAPES:
    at = ttnn.from_torch(torch.randn(*ash) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bt = ttnn.from_torch(torch.randn(*bsh) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m = 1
    for d in ash[:-1]:
        m *= d
    k, n = ash[-1], bsh[-1]
    gfl = 2 * m * k * n / 1e9
    row = {"m": m, "k": k, "n": n, "gflop": round(gfl, 2), "arms": {}}
    arms = [("default", None)]
    arms += [(f"{f}_f32acc1", ckc(f, True)) for f in ("HiFi4", "HiFi2", "LoFi")]
    arms += [(f"{f}_f32acc0", ckc(f, False)) for f in ("HiFi4", "HiFi2", "LoFi")]
    print(f"  {label}  M={m} K={k} N={n}  {gfl:.1f} GFLOP", flush=True)
    for name, kc in arms:
        def call(kc=kc):
            if kc is None:
                ttnn.deallocate(MM(at, bt, memory_config=DRAM, dtype=ttnn.bfloat16))
            else:
                ttnn.deallocate(MM(at, bt, memory_config=DRAM, dtype=ttnn.bfloat16,
                                   compute_kernel_config=kc))
        try:
            r = [gfl / (timed(dev, call) / 1e3) / 1e3 for _ in range(2)]
        except Exception as e:
            print(f"      {name:14s} ERR {str(e)[:60]}", flush=True)
            continue
        row["arms"][name] = [round(x, 2) for x in r]
        print(f"      {name:14s} {r[0]:7.2f} / {r[1]:7.2f} TFLOP/s  "
              f"(A/A {abs(r[0]-r[1])/max(r):.2%})", flush=True)
    res["trunk_knobs"][label] = row
    ttnn.deallocate(at); ttnn.deallocate(bt)

# ------------------------------------------------- D: N widening at fixed K=256, production ckc
print("\n== D. N widening at K=256, production ckc (HiFi4+f32acc) ==", flush=True)
kc4 = ckc("HiFi4", True)
at = ttnn.from_torch(torch.randn(1, 320, 320, 256) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
M = 320 * 320
res["n_widen_hifi4"] = {}
for n in (128, 256, 512, 768, 1024, 2048):
    bt = ttnn.from_torch(torch.randn(256, n) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gfl = 2 * M * 256 * n / 1e9
    try:
        r = [gfl / (timed(dev, lambda: ttnn.deallocate(
            MM(at, bt, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=kc4))) / 1e3) / 1e3
             for _ in range(2)]
        res["n_widen_hifi4"][n] = [round(x, 2) for x in r]
        print(f"  N={n:5d} {r[0]:7.2f} / {r[1]:7.2f} TFLOP/s  (A/A {abs(r[0]-r[1])/max(r):.2%})", flush=True)
    except Exception as e:
        print(f"  N={n:5d} ERR {str(e)[:70]}", flush=True)
    ttnn.deallocate(bt)
ttnn.deallocate(at)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], flush=True)
