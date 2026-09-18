#!/usr/bin/env python3
"""The two L1-path layout deletions, alone and stacked: parity, firing census, op-level A/B.

  back   -- the output channel move as ONE pass. Today: `ttnn.permute(0,2,3,1)` into L1 and then
            `ttnn.clone` to DRAM for the concat, two full passes over a tensor read once.
  gated  -- the fused forward move on the L1 path. `reblock_permute.eligible_gated` has always
            had an L1 clause; the call site asked for a DRAM destination on top of it, so the
            clause was unreachable and the L1 path kept `ttnn.chunk` + 2 `multiply_` + 2 plain
            channel moves where the kernel does all five in two passes.

Op level is a SCREEN. Four of this campaign's op-level levers transferred to the fold at
25x-to-infinite error with two flipping sign, so nothing here is a result until `fold_ab.py`
runs it against an A/A floor.

The negative control breaks the very thing the parity check reads: it swaps the back kernel for
a WRONG permutation of the right shape, and the trimul output has to move. A uniform mask scale
does NOT work as a control here -- the tail's layer_norm divides it straight back out, which is
how the first version of this check passed with a deliberately broken arm.
"""
import itertools
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
SIZES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                          else ["256", "288", "320", "512"])]

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

ARMS = [("base", False, False), ("back", True, False), ("gated", False, True),
        ("stack", True, True)]


def setarm(back, gated):
    tt.set_trimul_back_one_pass_l1(back)
    tt.set_trimul_gated_move_l1(gated)


def counters():
    return (RB.STATS_BACK[0], RB.STATS_GATED[0], RB.STATS[0])


def run(tm, z, mask, back, gated):
    setarm(back, gated)
    c0 = counters()
    out = tm(z, mask)
    r = ttnn.to_torch(out)
    ttnn.deallocate(out)
    return r, tuple(b - a for a, b in zip(c0, counters()))


def wall(tm, z, mask, reps=6):
    for _ in range(2):
        ttnn.deallocate(tm(z, mask))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.deallocate(tm(z, mask))
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e3 / reps


OUT = {"host": TAG, "reps": REPS, "grid": list(tt.COMPUTE_GRID_MAIN),
       "counters": "(back_kernel, gated_kernel, fwd_kernel) calls per trimul call",
       "parity": [], "ab": []}

with during(period=1.5) as clk:
    for N in SIZES:
        torch.manual_seed(1)
        z = f(torch.randn(1, N, N, 128))
        m1 = torch.ones(1, N)
        mask = f(m1[:, :, None] * m1[:, None, :])
        for name, tm in MODS.items():
            ref_t, c_base = run(tm, z, mask, False, False)
            for arm, bk, gt in ARMS[1:]:
                got, c = run(tm, z, mask, bk, gt)
                eq = torch.equal(ref_t, got)
                mx = float((ref_t.float() - got.float()).abs().max())
                OUT["parity"].append(dict(n=N, mod=name, arm=arm, torch_equal=eq, max_abs=mx,
                                          calls_base=list(c_base), calls_arm=list(c),
                                          fired=c != c_base))
                print(f"N={N:4d} {name:5s} {arm:6s} torch.equal={eq} max_abs={mx:g} "
                      f"calls {c_base}->{c} {'FIRED' if c != c_base else 'DID NOT FIRE'}")
        # --- negative control, once per size: a wrong permutation of the right shape
        tm = MODS["start"]
        good, _ = run(tm, z, mask, False, False)
        orig = RB.reblock_permute_back
        RB.reblock_permute_back = lambda x, mc=None, device=None: ttnn.permute(
            x, (0, 3, 2, 1), memory_config=mc or x.memory_config())
        bad, cnc = run(tm, z, mask, True, True)
        RB.reblock_permute_back = orig
        differs = not torch.equal(good, bad)
        OUT.setdefault("negative_control", []).append(dict(
            n=N, differs=differs, max_abs=float((good.float() - bad.float()).abs().max())))
        print(f"  neg-ctrl N={N} wrong-permute back move: output differs={differs} "
              f"max_abs={OUT['negative_control'][-1]['max_abs']:g}")
        # --- interleaved A/B: every arm in every rep, plus the base twice for the A/A floor
        for name, tm in MODS.items():
            acc = {a: [] for a, _, _ in ARMS}
            acc["base2"] = []
            for _ in range(REPS):
                for arm, bk, gt in ARMS:
                    setarm(bk, gt)
                    acc[arm].append(wall(tm, z, mask))
                setarm(False, False)
                acc["base2"].append(wall(tm, z, mask))
            med = {k: st.median(v) for k, v in acc.items()}
            floor = max(med["base"], med["base2"]) / min(med["base"], med["base2"])
            row = dict(n=N, mod=name, aa_floor=round(floor, 4),
                       **{f"{k}_ms": round(v, 4) for k, v in med.items()},
                       **{f"{a}_ratio": round(med["base"] / med[a], 4) for a, _, _ in ARMS[1:]})
            OUT["ab"].append(row)
            print(f"  A/B N={N:4d} {name:5s} base {med['base']:7.4f}  "
                  + "  ".join(f"{a} {med[a]:7.4f} ({med['base']/med[a]:6.4f}x)"
                              for a, _, _ in ARMS[1:])
                  + f"   A/A floor {floor:6.4f}x")
        ttnn.deallocate(z)
        ttnn.deallocate(mask)

OUT["clock"] = clk.summary()
print("\n" + clk.line(0))
p = HERE / f"l1_levers_{TAG}.json"
p.write_text(json.dumps(OUT, indent=2))
print("WROTE", p)
bad = [r for r in OUT["parity"] if not r["torch_equal"]]
nc_bad = [r for r in OUT["negative_control"] if not r["differs"]]
print("PARITY:", "ALL torch.equal" if not bad else f"BROKEN at {bad}")
print("NEG-CTRL:", "fires at every size" if not nc_bad else f"BLIND at {nc_bad}")
