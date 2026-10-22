"""gin_ragged's roof at the bytes it actually moves on today's main (qs 2 at 224), against its measured call time.

Card-free. Call times are bc4-bet-genroof's probe (qb2 chip 3, AICLK 1350, 7e48589fc; gin_ragged's kernels are
unchanged on main since). Prints p300c device s per BC2 gradient iteration and the x0.80 p150a reading.
    python3 -I perf/bc4_flops/gin_roof.py
"""
BW, RATE, P150A = 391e9, 45e12, 0.80
TB, S, K, UNITS = 2048, 224, 128, 98          # bf16 tile bytes, tokens, c_z, units at qs 2 (49 cubes x 2)
X_UNIT = 32 * (K // 32) * TB                   # each unit reads its whole 32-row x cube, all K
# (calls/it, out channels C, measured ms: full, no compute (abl 19), reads only (trikern, whole call))
SHAPES = {"whole C128": (104, 128, 0.3578, 0.3206, 0.140), "chunk C64": (208, 64, 0.2377, 0.2023, None)}
TRUE_BYTE_S, FLOP_S = 0.0245, 0.0307           # genroof ROOFS.md: x once, weights once, outputs once; matmul FLOPs

tot = {"dev": 0.0, "byte": 0.0, "flop": 0.0, "nocomp": 0.0}
for name, (n, C, ms, ms_nc, ms_rd) in SHAPES.items():
    w = 2 * (2 * C // 32) * (K // 32) * TB     # wp and wg, every unit
    rd = UNITS * (X_UNIT + w)
    wr = 2 * C * S * S * 2
    fl = 2 * S * S * K * 4 * C
    print(f"{name}: reads {rd/1e6:.1f} MB, writes {wr/1e6:.1f} MB -> {(rd+wr)/BW*1e3:.3f} ms; "
          f"{fl/1e9:.2f} GFLOP -> {fl/RATE*1e3:.3f} ms; measured {ms:.3f}, no compute {ms_nc:.3f}"
          + (f", reads only {ms_rd:.3f} = {rd/ms_rd/1e6:.0f} GB/s" if ms_rd else ""))
    tot["dev"] += n * ms / 1e3; tot["byte"] += n * (rd + wr) / BW; tot["flop"] += n * fl / RATE
    tot["nocomp"] += n * ms_nc / 1e3
    if ms_rd:
        rate_rd = rd / (ms_rd / 1e3)
byte_rd = tot["byte"] * BW / rate_rd
for k, v in tot.items():
    print(f"{k:7s} {v:.4f} s/it")
print(f"above max(true bytes, FLOPs) {tot['dev'] - max(TRUE_BYTE_S, FLOP_S):.4f} p300c "
      f"{(tot['dev'] - max(TRUE_BYTE_S, FLOP_S)) * P150A:.4f} p150a")
print(f"above max(qs-2 bytes, FLOPs) {tot['dev'] - max(tot['byte'], tot['flop']):.4f} p300c "
      f"{(tot['dev'] - max(tot['byte'], tot['flop'])) * P150A:.4f} p150a")
print(f"qs-2 bytes at gin's own read rate {rate_rd/1e9:.0f} GB/s: {byte_rd:.4f} s/it; "
      f"full-overlap ceiling {tot['dev'] - byte_rd:.4f} p300c {(tot['dev'] - byte_rd) * P150A:.4f} p150a")
print(f"all compute removed: {tot['dev'] - tot['nocomp']:.4f} p300c {(tot['dev'] - tot['nocomp']) * P150A:.4f} p150a")
