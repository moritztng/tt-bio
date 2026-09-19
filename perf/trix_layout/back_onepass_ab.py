#!/usr/bin/env python3
"""The L1 path's output channel move as ONE pass: parity, firing census, op-level A/B.

Today the L1 path writes `ttnn.permute(0,2,3,1)` into L1 and then clones the chunk to DRAM for
the concat -- two full passes over a tensor read once. `reblock_permute_back` takes the DRAM
destination itself where `eligible_back` serves, so the clone drops out.

Three things, in this order, because a ratio with no parity behind it is worth nothing:
  1. `torch.equal` on the whole trimul output, arm against arm, at every N in the window;
  2. the firing census, so a dead flag cannot pass as a correct one -- and a negative control
     that breaks the very thing the check reads;
  3. an INTERLEAVED op-level A/B with an A/A floor, clock sampled DURING.

Op level is a SCREEN. The fold A/B is `fold_ab.py`.
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from clocksample import during  # noqa: E402

import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__, f"wrong tt_bio: {tt_bio.__file__}"
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import reblock_permute as RB  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
# 256/288/320/352 are the L1 window; 298 is ragged and must decline; 512 is the DRAM path and
# must be untouched. A lever that changes a shape it does not claim is a defect, not a win.
SIZES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                          else ["256", "288", "320", "352", "298", "512"])]

dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
for k, v in list(weights.items()):
    if v.numel() and float(v.abs().max()) == 0.0:
        weights[k] = torch.randn_like(v) * 0.05
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)
MODS = {"start": layer.triangle_multiplication_start,
        "end": layer.triangle_multiplication_end}

f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def arm(tm, z, mask, on):
    tt.set_trimul_back_one_pass_l1(on)
    before = RB.STATS_BACK[0]
    out = tm(z, mask)
    r = ttnn.to_torch(out)
    ttnn.deallocate(out)
    return r, RB.STATS_BACK[0] - before


def wall(tm, z, mask, reps=6):
    for _ in range(2):
        ttnn.deallocate(tm(z, mask))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(tm(z, mask))
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e3 / reps


OUT = {"host": TAG, "reps": REPS, "grid": list(tt.COMPUTE_GRID_MAIN), "parity": [], "ab": []}

with during(period=1.5) as clk:
    for N in SIZES:
        torch.manual_seed(1)
        z = f(torch.randn(1, N, N, 128))
        m1 = torch.ones(1, N)
        mask = f(m1[:, :, None] * m1[:, None, :])
        for name, tm in MODS.items():
            off, n_off = arm(tm, z, mask, False)
            on, n_on = arm(tm, z, mask, True)
            eq = torch.equal(off, on)
            mx = float((off.float() - on.float()).abs().max())
            row = dict(n=N, mod=name, torch_equal=eq, max_abs=mx,
                       back_kernel_calls_off=n_off, back_kernel_calls_on=n_on,
                       fired=n_on > n_off)
            OUT["parity"].append(row)
            print(f"N={N:4d} {name:5s} torch.equal={eq} max_abs={mx:g} "
                  f"back-kernel calls off={n_off} on={n_on} "
                  f"{'FIRED' if n_on > n_off else 'did not fire'}")
        # --- negative control: break what the parity check reads, and watch it fail.
        if N == SIZES[0]:
            tm = MODS["start"]
            good, _ = arm(tm, z, mask, False)
            prev = RB.set_enabled_back(False)
            bad_mask = f((m1[:, :, None] * m1[:, None, :]) * 0.5)
            bad, _ = arm(tm, z, bad_mask, True)
            RB.set_enabled_back(prev)
            nc = not torch.equal(good, bad)
            OUT["negative_control"] = dict(n=N, differs=nc,
                                           max_abs=float((good.float() - bad.float()).abs().max()))
            print(f"negative control (halved mask): differs={nc} "
                  f"max_abs={OUT['negative_control']['max_abs']:g}")
            ttnn.deallocate(bad_mask)
        # --- interleaved op-level A/B with an A/A floor
        for name, tm in MODS.items():
            a, b, aa = [], [], []
            for _ in range(REPS):
                tt.set_trimul_back_one_pass_l1(False)
                a.append(wall(tm, z, mask))
                tt.set_trimul_back_one_pass_l1(True)
                b.append(wall(tm, z, mask))
                tt.set_trimul_back_one_pass_l1(False)
                aa.append(wall(tm, z, mask))
            ma, mb, maa = st.median(a), st.median(b), st.median(aa)
            row = dict(n=N, mod=name, off_ms=round(ma, 4), on_ms=round(mb, 4),
                       off2_ms=round(maa, 4), ratio=round(ma / mb, 4),
                       aa_floor=round(max(ma, maa) / min(ma, maa), 4))
            OUT["ab"].append(row)
            print(f"  A/B N={N:4d} {name:5s} off {ma:7.4f}  on {mb:7.4f}  "
                  f"{ma / mb:6.4f}x   A/A floor {row['aa_floor']:6.4f}x")
        ttnn.deallocate(z)
        ttnn.deallocate(mask)

OUT["clock"] = clk.summary()
print("\n" + clk.line(0))
p = HERE / f"back_onepass_{TAG}.json"
p.write_text(json.dumps(OUT, indent=2))
print("WROTE", p)
bad = [r for r in OUT["parity"] if not r["torch_equal"]]
print("PARITY:", "ALL torch.equal" if not bad else f"BROKEN at {bad}")
