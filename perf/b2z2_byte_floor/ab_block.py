"""Paired interleaved A/B of one PairformerLayer with the fused qkv+gate projection on and off.

Both currencies side by side, which is the whole point of this row: the arm deletes a measured
number of bytes and the block moves by a measured ratio, and the two are reported next to each
other rather than one being inferred from the other.

One process, one device, one built layer, arms alternating base/lever/base/lever so a drift in the
card cannot land entirely on one arm. An A/A leg (base against base, same interleaving) sets the
floor the ratio has to clear. The eligibility census of every fused kernel is printed per arm,
because a byte lever that quietly drops a tuned kernel is how the previous byte row's sign flipped.
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
OUT = os.environ.get("AB_OUT", f"/tmp/b2z2_bytefloor_ab_{S}.json")

dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)

s = torch.randn(1, S, 384)
z = torch.randn(1, S, S, 128)
m1 = torch.ones(1, S)
pair_mask = m1[:, :, None] * m1[:, None, :]
attn = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
mask_tt, attn_tt = f(pair_mask), f(attn)


def census():
    return {"qkvg": list(K.QKVG_STATS), "qkv_heads": list(K.STATS), "tail": list(K.TAIL_STATS),
            "reblock_gated": int(RP.STATS_GATED[0])}


def one(on):
    """One timed block. `z` is updated in place, so both the input and the output are fresh."""
    K._QKVG_ENABLED = bool(on)
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

res = {"tokens": S, "reps": REPS, "arms": {}}
for leg, arms in (("AB", (False, True)), ("AA", (False, False))):
    t = {0: [], 1: []}
    digs = {0: set(), 1: set()}
    c0 = census()
    for r in range(REPS):
        for i, on in enumerate(arms):
            dt, dig = one(on)
            t[i].append(dt)
            digs[i].add(round(dig, 3))
    c1 = census()
    a, b = statistics.median(t[0]), statistics.median(t[1])
    res["arms"][leg] = {
        "base_ms": a, "lever_ms": b, "ratio": a / b,
        "base_all": t[0], "lever_all": t[1],
        "base_spread_pct": 100 * (max(t[0]) - min(t[0])) / a,
        "lever_spread_pct": 100 * (max(t[1]) - min(t[1])) / b,
        "digest_base": sorted(digs[0]), "digest_lever": sorted(digs[1]),
        "census_delta": {k: [c1[k][i] - c0[k][i] for i in range(len(c1[k]))]
                         if isinstance(c1[k], list) else c1[k] - c0[k] for k in c1},
    }
    print(f"[{leg}] base {a:8.3f} ms   arm {b:8.3f} ms   ratio {a/b:.5f}   "
          f"spread {res['arms'][leg]['base_spread_pct']:.2f}/"
          f"{res['arms'][leg]['lever_spread_pct']:.2f} %", flush=True)
    print(f"     census delta {res['arms'][leg]['census_delta']}", flush=True)
    print(f"     block sum base {sorted(digs[0])}  arm {sorted(digs[1])}", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("->", OUT)
