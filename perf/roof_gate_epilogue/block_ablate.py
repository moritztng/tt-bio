#!/usr/bin/env python3
"""The gate multiply's price on a whole Pairformer block, by deleting it.

The fusion cannot be built in the producer (the multiply's other operand is downstream of the
producer's own output), so the block A/B here is an ABLATION, not a lever: arm `nogate` elides
`gate_and_project`'s `multiply_` outright, which is numerically wrong and is exactly the ceiling a
correct fusion could reach if the epilogue cost nothing. The realistic number is that ceiling minus
the 67.1 MB read the hosting kernel gains, priced at the multiply's own measured bandwidth.

Arms interleave base/nogate/base/nogate, with an A/A leg for the floor. The shim fires only on the
gate signature -- both operands 4-D `[S, H, S, 32]` with a SIGMOID activation on b -- so no other
`multiply_` in the block is touched, and it counts its own hits.
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
OUT = os.environ.get("AB_OUT", f"perf/roof_gate_epilogue/block_ablate_{S}_whglx_c2.json")

dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

# --- the ablation shim -----------------------------------------------------------------------
_real_mul_ = ttnn.multiply_
HITS = [0, 0]          # [triatt gate multiplies seen, elided]
SHAPES = {}            # every sigmoid-gated multiply_ in the block, by shape, so the shim's reach
ELIDE = [False]        # is a printed census and not an assumption


def _shim(a, b, *args, **kw):
    acts = kw.get("input_tensor_b_activations")
    sh = list(getattr(a, "shape", []))
    if acts is not None and len(sh) == 4 and sh == list(getattr(b, "shape", [])):
        SHAPES[tuple(sh)] = SHAPES.get(tuple(sh), 0) + 1
        # the triangle-attention gate, head-major [S, n_heads, S, 32]: batch axis is the token
        # axis (not 1, which is the trimul input gate) and the last axis is one tile
        if sh[0] != 1 and sh[-1] == 32:
            HITS[0] += 1
            if ELIDE[0]:
                HITS[1] += 1
                ttnn.deallocate(b)
                return a
    return _real_mul_(a, b, *args, **kw)


ttnn.multiply_ = _shim
tt.ttnn.multiply_ = _shim

torch.manual_seed(0)
rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
weights = {k: v.float() for k, v in rl.state_dict().items()}
# `ref.PairformerLayer` zero-initialises every output projection -- tri_att `linear_o`/`linear_g`,
# tri_mul `p_out`/`g_out`, `transition_*.fc3`, `attention.proj_o` -- so at init the block is the
# identity on z and ANY digest taken from its output is a constant. The campaign's shared block
# harness (`perf/b2z2_byte_floor/ab_block.py`) reads `out_z.sum()` off exactly this construction,
# which makes its bit-exactness control blind to the whole triangle-attention tail. Verified on
# whglx card 2: eliding both gate multiplies leaves `torch.equal(out_z_gated, out_z_elided)` True
# with max|dz| = 0.0. Refill the zero 2-D weights so the digest can move.
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
            "sdpa_fused": list(TS.STATS), "gate_mul_seen": HITS[0],
            "gate_mul_elided": HITS[1],
            "gated_mul_shapes": {str(kk): vv for kk, vv in SHAPES.items()}}


def one(elide):
    ELIDE[0] = bool(elide)
    s_tt, z_tt = f(s), f(z)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out_s, out_z = layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1e3
    zt = ttnn.to_torch(out_z).float()
    dig = (float(zt.sum()), float(zt.abs().sum()))
    ttnn.deallocate(out_z)
    ttnn.deallocate(out_s)
    return dt, dig


for e in (False, True):                       # compile both arms before anything is timed
    one(e)

res = {"tokens": S, "reps": REPS, "refilled_zero_weights": ZEROED, "arch": str(dev.arch()), "host": os.uname().nodename,
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg": os.getloadavg(),
       "grid": list(tt.COMPUTE_GRID_MAIN), "arms": {}}
for leg, arms in (("AB", (False, True)), ("AA", (False, False))):
    t = {0: [], 1: []}
    digs = {0: set(), 1: set()}
    c0 = census()
    for r in range(REPS):
        for i, e in enumerate(arms):
            dt, dig = one(e)
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
    print(leg, json.dumps({kk: res["arms"][leg][kk] for kk in ("a_ms", "b_ms", "ratio")}), flush=True)

ab, aa = res["arms"]["AB"], res["arms"]["AA"]
res["derived"] = {
    "block_ms": ab["a_ms"],
    "ceiling_ratio_block": ab["ratio"],
    "aa_floor_frac": abs(aa["ratio"] - 1.0),
    "saved_ms_ceiling": ab["a_ms"] - ab["b_ms"],
}
print(json.dumps(res["derived"], indent=1), flush=True)
p = REPO / OUT
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res, indent=1))
print("wrote", p, flush=True)
