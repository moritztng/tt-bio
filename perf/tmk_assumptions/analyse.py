#!/usr/bin/env python3
"""Turn rate_gap.json into the binding-roof table. CPU only, no device.

A TFLOP/s gap is headroom only if the shape's arithmetic intensity says it can be reached, so
every call site is scored against min(compute roof, its own traffic roof at ITS OWN read:write
mix) -- never against the square cube.
"""
import json, sys

d = json.load(open(sys.argv[1]))
A = d["arms"]


def g(name, k="serial_TFLOPs"):
    return A[name][k]


# roofs, in-session
cube = max(A["roof_cube4096"]["serial_TFLOPs"], A["roof_cube4096_AA"]["serial_TFLOPs"])
bw = {
    "2R1W": max(A["roof_dram_2R1W"]["serial_GBs"], A["roof_dram_2R1W_AA"]["serial_GBs"]),
    "1R1W": max(A["roof_dram_1R1W"]["serial_GBs"], A.get("roof_dram_clone", A["roof_dram_1R1W"])["serial_GBs"]),
    "1R2W": A["roof_dram_1R2W"]["serial_GBs"],
    "1R0W": A.get("roof_dram_1R0W", {}).get("serial_GBs"),
}
# A per-byte cost model from the two clean endpoints, so a mix the harness has no clean op for is
# interpolated rather than guessed. r + w = 1 by construction of each arm's byte split.
#   1R1W: (1/2)/BWr + (1/2)/BWw = 1/BW_1R1W ; 2R1W: (2/3)/BWr + (1/3)/BWw = 1/BW_2R1W
import numpy as np
M = np.array([[0.5, 0.5], [2 / 3, 1 / 3]])
rhs = np.array([1 / bw["1R1W"], 1 / bw["2R1W"]])
inv_r, inv_w = np.linalg.solve(M, rhs)
BWr, BWw = 1 / inv_r, 1 / inv_w


def mix_roof(nr, nw):
    """GB/s for a stream that is nr read bytes to nw write bytes."""
    f = nr + nw
    return 1.0 / ((nr / f) * inv_r + (nw / f) * inv_w)


print(f"in-session roofs: cube {cube:.2f} TFLOP/s | 2R1W {bw['2R1W']:.1f} | "
      f"1R1W {bw['1R1W']:.1f} | 1R2W(concat) {bw['1R2W']:.1f} | read-only {bw['1R0W']} GB/s")
print(f"solved per-direction: read {BWr:.1f} GB/s, write {BWw:.1f} GB/s  "
      f"-> machine balance at the cube = {cube * 1e12 / (bw['2R1W'] * 1e9):.0f} FLOP/byte (2R1W)")
print()

SITES = [
    ("A_inproj_DRAM",   "in-projection  [1,1,262144,256]@[256,512]", 1, 2),
    ("B_contract_DRAM", "contraction    [1,128,512,512]^2",          2, 1),
    ("C_outproj_linear", "out-projection [1,512,512,256]@[256,256]", 1, 1),
]
hdr = f"{'site':46s} {'TF/s':>7s} {'GB/s':>7s} {'AI':>7s} {'trafroof':>9s} {'bind':>8s} {'%bind':>7s} {'head':>6s}"
print(hdr); print("-" * len(hdr))
tot = {}
for key, label, nr, nw in SITES:
    a = A[key]
    r = mix_roof(nr, nw)
    traf = a["AI_flop_per_byte"] * r / 1e3           # TFLOP/s
    bind = min(cube, traf)
    tot[key] = dict(rate=a["serial_TFLOPs"], bind=bind, ms=a["serial_ms"], GFLOP=a["GFLOP"])
    print(f"{label:46s} {a['serial_TFLOPs']:7.2f} {a['serial_GBs']:7.1f} "
          f"{a['AI_flop_per_byte']:7.1f} {traf:9.2f} {bind:8.2f} "
          f"{100 * a['serial_TFLOPs'] / bind:6.1f}% {bind / a['serial_TFLOPs']:5.2f}x")
print()
print(f"{'campaign framing: % of the 123.65 cube and the implied headroom':60s}")
for key, label, nr, nw in SITES:
    a = A[key]
    print(f"  {label:46s} {100 * a['serial_TFLOPs'] / 123.65:5.1f}% of cube -> "
          f"{123.65 / a['serial_TFLOPs']:.2f}x claimed, "
          f"{tot[key]['bind'] / a['serial_TFLOPs']:.2f}x real")
print()
for k in ("A_inproj", "B_contract", "C_outproj"):
    base = {"A_inproj": "A_inproj_DRAM", "B_contract": "B_contract_DRAM",
            "C_outproj": "C_outproj_linear"}[k]
    b8 = k + "_bfp8"
    if b8 in A:
        print(f"bfp8_b {k:12s} {A[base]['serial_ms']:.4f} -> {A[b8]['serial_ms']:.4f} ms  "
              f"{A[base]['serial_ms'] / A[b8]['serial_ms']:.3f}x   "
              f"bytes {A[base]['MB']:.1f} -> {A[b8]['MB']:.1f} MB "
              f"({A[base]['MB'] / A[b8]['MB']:.3f}x)  "
              f"{A[b8]['serial_TFLOPs']:.2f} TFLOP/s, {A[b8]['serial_GBs']:.1f} GB/s")
print()
for k, v in A.items():
    if k.startswith("G_"):
        print(f"granule {k:24s} {v['serial_ms']:8.4f} ms  {v['serial_GBs']:7.1f} GB/s  {v['note']}")
print()
print("clock:", json.dumps(d["clock"]))
