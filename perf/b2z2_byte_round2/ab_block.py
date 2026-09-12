"""Paired interleaved A/B of one PairformerLayer with one or more read-deleting levers on and off.

One process, one device, one built layer, arms alternating base/lever/base/lever so a drift in the
card cannot land entirely on one arm. An A/A leg (base against base, same interleaving) sets the
floor the ratio has to clear. `os.getloadavg()` is recorded INSIDE every rep, because whglx runs
several rows at once and a one-shot check before the loop is blind to a neighbour that starts
mid-run (CONTEXT 4-I).

The fused-kernel eligibility census is printed per arm: a byte lever that quietly drops a tuned
kernel is how an earlier byte row's sign flipped.

The fixture is UNZEROED. `tt_bio.reference` zero-initialises every sub-unit's output projection, so
with `p_out.weight` zero the trimul's whole contribution is zero and the block's output digest
cannot see it. Shapes and ops are untouched by this, so the timing is the timing either way.

  LEVERS=gout,qkvg AB_REPS=7 python3 ab_block.py
"""
import json
import os
import statistics
import sys
import time

import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import triatt_qkv as K
from tt_bio import reference as ref
from tt_bio import reblock_permute as RP

S = int(os.environ.get("AB_TOKENS", "512"))
REPS = int(os.environ.get("AB_REPS", "7"))
LEVERS = [x for x in os.environ.get("LEVERS", "gout").split(",") if x]
OUT = os.environ.get("AB_OUT", f"/tmp/b2z2_round2_ab_{'_'.join(LEVERS)}_{S}.json")

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

s = torch.randn(1, S, 384)
z = torch.randn(1, S, S, 128)
m1 = torch.ones(1, S)
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
mask_tt, attn_tt = f(m1[:, :, None] * m1[:, None, :]), f((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)

SET = {"gout": tt.set_trimul_fused_gout,
       "qkvg": lambda on: setattr(K, "_QKVG_ENABLED", bool(on))}
for name in LEVERS:
    if name not in SET:
        raise SystemExit(f"unknown lever {name!r}; known: {sorted(SET)}")


def census():
    return {"trimul_gout": list(tt.TRIMUL_GOUT_STATS), "qkvg": list(K.QKVG_STATS),
            "qkv_heads": list(K.STATS), "tail": list(K.TAIL_STATS),
            "trimul_tail_f1": list(tt._trimul_tail.STATS),
            "reblock_gated": int(RP.STATS_GATED[0])}


def one(on):
    """One timed block. `z` is re-uploaded every call, so the input is always the same numbers."""
    for name in LEVERS:
        SET[name](on)
    s_tt, z_tt = f(s), f(z)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out_s, out_z = layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1e3
    dig = float(ttnn.to_torch(out_z).float().sum())
    ttnn.deallocate(out_z)
    ttnn.deallocate(out_s)
    return dt, dig


for on in (False, True):                    # compile both arms before anything is timed
    one(on)

res = {"tokens": S, "reps": REPS, "levers": LEVERS, "arms": {}}
for leg, arms in (("AB", (False, True)), ("AA", (False, False))):
    t = {0: [], 1: []}
    digs = {0: set(), 1: set()}
    load = []
    c0 = census()
    for r in range(REPS):
        for i, on in enumerate(arms):
            load.append(round(os.getloadavg()[0], 2))
            dt, dig = one(on)
            t[i].append(dt)
            digs[i].add(round(dig, 3))
    c1 = census()
    a, b = statistics.median(t[0]), statistics.median(t[1])
    res["arms"][leg] = {
        "base_ms": a, "lever_ms": b, "ratio": a / b,
        "base_all": t[0], "lever_all": t[1], "loadavg_per_rep": load,
        "base_spread_pct": 100 * (max(t[0]) - min(t[0])) / a,
        "lever_spread_pct": 100 * (max(t[1]) - min(t[1])) / b,
        "digest_base": sorted(digs[0]), "digest_lever": sorted(digs[1]),
        "census_delta": {k: [c1[k][i] - c0[k][i] for i in range(len(c1[k]))]
                         if isinstance(c1[k], list) else c1[k] - c0[k] for k in c1},
    }
    print(f"[{leg}] base {a:8.3f} ms   arm {b:8.3f} ms   ratio {a/b:.5f}   "
          f"spread {res['arms'][leg]['base_spread_pct']:.2f}/"
          f"{res['arms'][leg]['lever_spread_pct']:.2f} %   loadavg "
          f"{min(load):.1f}-{max(load):.1f}", flush=True)
    print(f"     census delta {res['arms'][leg]['census_delta']}", flush=True)
    print(f"     block sum base {sorted(digs[0])}  arm {sorted(digs[1])}", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("->", OUT)
