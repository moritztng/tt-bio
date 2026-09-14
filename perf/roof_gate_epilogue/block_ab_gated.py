#!/usr/bin/env python3
"""The gate epilogue's price on a whole Pairformer block, as a LEVER and not an ablation.

`block_ablate.py` next to this file measured the ceiling by eliding the gate multiply outright,
which is numerically wrong on purpose. This is the built thing: arm A is the shipped pair (fused
SDPA, then `gate_and_project`'s `multiply_`), arm B is the same block with
`TT_BIO_TRIATT_GATE_EPILOGUE` on, so the multiply program is gone and the kernel does it. Both arms
compute the same answer, so the digest is a real control here rather than a deliberate mismatch.

Arms interleave A/B/A/B per rep and an A/A leg gives the floor
(`op-ab-must-interleave-arms-compile-warmup-bias`). The census proves the lever fired: `gate_served`
counts the calls the kernel took the gate on and `gate_mul` counts the `multiply_` programs that
still ran.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as tt
from tt_bio import triatt_qkv as K
from tt_bio import reference as ref
from tt_bio import triatt_sdpa as TS

S = int(os.environ.get("AB_TOKENS", "512"))
REPS = int(os.environ.get("AB_REPS", "7"))
OUT = os.environ.get("AB_OUT", f"perf/roof_gate_epilogue/block_ab_gated_{S}_qb2_c2.json")

dev = tt.get_device()
KCLS = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
KC = KCLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
          fp32_dest_acc_en=True, packer_l1_acc=True)

# Count the gate multiplies that STILL run. With the lever on this must fall to the trimul input
# gates only -- if it does not, the epilogue declined and the ratio means nothing.
_real_mul_ = ttnn.multiply_
MUL = {}


def _shim(a, b, *args, **kw):
    acts = kw.get("input_tensor_b_activations")
    sh = list(getattr(a, "shape", []))
    if acts is not None and len(sh) == 4 and sh == list(getattr(b, "shape", [])):
        MUL[tuple(sh)] = MUL.get(tuple(sh), 0) + 1
    return _real_mul_(a, b, *args, **kw)


ttnn.multiply_ = _shim
tt.ttnn.multiply_ = _shim

torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
# `ref.PairformerLayer` zero-initialises every output projection, so at init the block is the
# identity on z and any digest off its output is a constant -- verified on whglx card 2 by
# `block_ablate.py`, which is why it refills them and `perf/b2z2_byte_floor/ab_block.py` cannot
# see a triangle-attention change at all. Refill so the digest can move.
ZEROED = sorted(k for k, v in weights.items() if v.dim() == 2 and float(v.abs().max()) == 0.0)
for k in ZEROED:
    v = weights[k]
    weights[k] = torch.randn(v.shape) * (v.shape[-1] ** -0.5)
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
            "sdpa_fused": list(TS.STATS), "gate_served": list(TS.GATE_STATS),
            "gate_rejects": {str(kk): vv for kk, vv in TS.GATE_REJECTS.items()},
            "sigmoid_mul_by_shape": {str(kk): vv for kk, vv in MUL.items()}}


def one(gated):
    TS._GATE_EPILOGUE = bool(gated)
    s_tt, z_tt = f(s), f(z)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out_s, out_z = layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1e3
    zt = ttnn.to_torch(out_z).float()
    dig = (float(zt.sum()), float(zt.abs().sum()))
    exact = zt.clone()
    ttnn.deallocate(out_z)
    ttnn.deallocate(out_s)
    return dt, dig, exact


for g in (False, True):                       # compile both arms before anything is timed
    one(g)

# bit-exactness of the block output, taken once outside the timing loop
_, _, za = one(False)
_, _, zb = one(True)
BITEX = bool(torch.equal(za, zb))
DZ = float((za - zb).abs().max())
DREL = float((za - zb).pow(2).mean().sqrt() / za.pow(2).mean().sqrt())
del za, zb

res = {"tokens": S, "reps": REPS, "refilled_zero_weights": ZEROED, "arch": str(dev.arch()),
       "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "loadavg": os.getloadavg(), "grid": list(tt.COMPUTE_GRID_MAIN),
       "block_bit_exact": BITEX, "block_max_abs_dz": DZ, "block_rel_rms_dz": DREL,
       "arms": {}}
print("block bit-exact:", BITEX, "max|dz|", DZ, "rel_rms", DREL, flush=True)

for leg, arms in (("AB", (False, True)), ("AA", (False, False))):
    t = {0: [], 1: []}
    digs = {0: set(), 1: set()}
    c0 = census()
    for r in range(REPS):
        for i, g in enumerate(arms):
            dt, dig, _ = one(g)
            t[i].append(dt)
            digs[i].add(tuple(round(x, 3) for x in dig))
    res["arms"][leg] = {
        "a_ms": st.median(t[0]), "b_ms": st.median(t[1]),
        "ratio": st.median(t[0]) / st.median(t[1]),
        "a_all": t[0], "b_all": t[1],
        "a_digests": sorted(str(x) for x in digs[0]),
        "b_digests": sorted(str(x) for x in digs[1]),
        "census_before": c0, "census_after": census(),
    }
    print(leg, json.dumps({kk: res["arms"][leg][kk] for kk in ("a_ms", "b_ms", "ratio")}),
          flush=True)

ab, aa = res["arms"]["AB"], res["arms"]["AA"]
res["derived"] = {
    "block_ms": ab["a_ms"],
    "gated_ms": ab["b_ms"],
    "ratio_block": ab["ratio"],
    "aa_floor_frac": abs(aa["ratio"] - 1.0),
    "saved_ms": ab["a_ms"] - ab["b_ms"],
}
print(json.dumps(res["derived"], indent=1), flush=True)
print(json.dumps(census(), indent=1), flush=True)
p = REPO / OUT
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res, indent=1))
print("wrote", p, flush=True)
