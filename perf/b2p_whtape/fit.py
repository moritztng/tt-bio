"""Fit BindCraft 2's device peak to a + b*N^2 on both boards, from footprint rungs already measured.

Wormhole: b2p-wh fp1 (`.107` chip 30). Blackhole: b2p-ceiling footprint legs (qb1 p150a).
Card-free; reads nothing but the numbers below, copied from those rows' rung.json files.
"""
import numpy as np

WH = {192: 2.2377, 288: 4.1187, 352: 5.8155, 416: 7.843, 448: 8.9806, 480: 10.1995, 512: 11.5064}
BH = {544: 13.00, 608: 16.01, 736: 23.07, 832: 29.18, 864: 31.39}
USABLE_WH_GB = 11.5  # 512 held 11.51 GB and completed; 544 died with 0.547 GB free in 27 MB pieces

for name, d in (("wormhole", WH), ("blackhole", BH)):
    n = np.array(list(d), float)
    y = np.array(list(d.values()))
    A = np.vstack([np.ones_like(n), n ** 2]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"{name}: peak = {a:.3f} GB + {b * 1e6:.2f} KB/pair ({b * 1e9 / 256:.1f} bf16 [N,N,128] "
          f"tensors), max residual {max(abs(A @ [a, b] - y)):.3f} GB")
    if name == "wormhole":
        for t in (544, 576, 608, 640, 704, 768):
            allowed = (USABLE_WH_GB - a) / t ** 2
            print(f"  {t}: predicted {a + b * t * t:.2f} GB; allowed {allowed * 1e6:.1f} KB/pair, "
                  f"remove {(b - allowed) * 1e9 / 256:.0f} tensors")
