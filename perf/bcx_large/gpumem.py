#!/usr/bin/env python3
"""What the SAME gradient step costs in GPU memory. DERIVED, not measured: we own no NVIDIA card.

The step is the one `ladder.py` runs on the card: AF2 model_1_ptm, 4 extra-MSA + 48 Evoformer
blocks, one MSA row, taped and rematerialised per block, bf16 activations. That is also what
BindCraft 2's optimiser differentiates on a GPU: ColabDesign's `af/model.py` sets
`global_config.use_remat = True` and `global_config.bfloat16 = True` by default, and
`alphafold/model/modules.py` wraps each Evoformer and extra-MSA block in `hk.remat` inside a
`layer_stack`. So the GPU program has the same shape as ours -- per-block checkpointing, bf16 --
and the comparison is like for like.

Under remat, live memory at the worst moment is

    (a) the block-boundary activations kept for all 52 blocks, plus
    (b) the intermediates of the ONE block being recomputed inside its own backward, plus
    (c) that block's incoming cotangents, plus
    (d) parameters and their gradients.

Every tensor below is named with the line of `tt_bio/af2_reference.py` that produces it, and the
count is deliberately a LOWER bound for the GPU: where XLA can plausibly fuse or drop a tensor
(the pre-softmax logits, the layernorm's own scratch, the gate before it multiplies) it is left
out, and elementwise sums are assumed in place. A lower bound on their side is the conservative
direction for a claim that our side holds something theirs does not.

  python3 perf/bcx_large/gpumem.py --n 512 1024 1536 --card 80 141
"""
from __future__ import annotations

import argparse
import json

GB = 1e9
C_M, C_Z, C_EXTRA = 256, 128, 64          # af2_reference.py:46-49
TRI_HIDDEN, PAIR_HEADS, PAIR_HEAD_DIM = 128, 4, 32   # af2_reference.py:366
MSA_HEADS, OPM_HIDDEN, FACTOR = 8, 32, 4             # af2_reference.py:366-370
N_EXTRA_BLOCKS, N_EVO_BLOCKS = 4, 48                 # af2.py stack sizes
BYTES = 2                                            # bf16, both sides


def tri_mul(n):
    """TriangleMultiplication, af2_reference.py:236-248. Two per block."""
    h = TRI_HIDDEN
    return [("norm_in(z)", n * n * C_Z), ("p_in(x) masked+gated", n * n * 2 * h),
            ("sigmoid(g_in(x))", n * n * 2 * h), ("einsum ijc", n * n * h),
            ("norm_out", n * n * h), ("sigmoid(g_out(x))", n * n * C_Z)]


def tri_att(n):
    """TriangleAttention, af2_reference.py:259-266 through Attention._attend:157-173."""
    h, d = PAIR_HEADS, PAIR_HEAD_DIM
    return [("layer_norm(z)", n * n * C_Z), ("q", n * n * h * d), ("k", n * n * h * d),
            ("v", n * n * h * d), ("nonbatched bias linear(x)", n * n * h),
            ("softmax weights [n,h,n,n]", n * h * n * n),
            ("attn out [n,n,h,d]", n * n * h * d), ("sigmoid(linear_g)", n * n * h * d)]


def pair_transition(n):
    """ReluTransition(c_z, factor=4)."""
    return [("transition hidden", n * n * C_Z * FACTOR)]


def msa_track(n, c_m, s=1):
    """MSA row attention with pair bias + column attention + transition, s MSA rows."""
    h = MSA_HEADS
    return [("msa layer_norm", s * n * c_m), ("msa pair_norm(z)", n * n * C_Z),
            ("row bias linear(z) [n,n,h]", n * n * h),
            ("row q/k/v", 3 * s * n * c_m), ("row weights [s,h,n,n]", s * h * n * n),
            ("row out+gate", 2 * s * n * c_m),
            ("col q/k/v", 3 * s * n * c_m), ("col weights [n,h,s,s]", n * h * s * s),
            ("col out+gate", 2 * s * n * c_m),
            ("msa transition hidden", s * n * c_m * FACTOR)]


def opm(n, c_m, s=1):
    """OuterProductMean(c_m, hidden=32, c_z), af2_reference.py:268-289."""
    return [("opm norm", s * n * c_m), ("opm a", s * n * OPM_HIDDEN),
            ("opm b", s * n * OPM_HIDDEN),
            ("opm outer [n,n,h*h]", n * n * OPM_HIDDEN * OPM_HIDDEN)]


def block(n, c_m, s=1):
    t = msa_track(n, c_m, s) + opm(n, c_m, s)
    for _ in range(2):
        t += tri_mul(n)
    for _ in range(2):
        t += tri_att(n)
    t += pair_transition(n)
    t += [("residual carries (5 pair, 3 msa)", 5 * n * n * C_Z + 3 * s * n * c_m)]
    return t


def boundary(n, s=1):
    """What a remat'd stack keeps per block: the block's own inputs."""
    per_extra = n * n * C_Z + s * n * C_EXTRA
    per_evo = n * n * C_Z + s * n * C_M
    return N_EXTRA_BLOCKS * per_extra + N_EVO_BLOCKS * per_evo


PARAMS = 93_247_918          # AF2 model_1_ptm parameter count, params_model_1_ptm.npz


def report(n, s=1):
    ev = block(n, C_M, s)
    ex = block(n, C_EXTRA, s)
    worst_name, worst = ("evoformer", ev) if sum(v for _, v in ev) >= sum(v for _, v in ex) else ("extra_msa", ex)
    live_block = sum(v for _, v in worst)
    out = {
        "n": n, "msa_rows": s,
        "boundary_gb": boundary(n, s) * BYTES / GB,
        "block": worst_name,
        "block_recompute_gb": live_block * BYTES / GB,
        # A cotangent per live intermediate is the PESSIMISTIC end and is not in the headline:
        # reverse mode frees each gradient as it is consumed, so the true peak sits between the
        # two largest live cotangents and a full second copy of the tape. The lower end is the
        # GPU-friendly one, so that is what `total_gb` carries and what the claim is made on.
        "cotangent_low_gb": 2 * max(v for _, v in worst) * BYTES / GB,
        "cotangent_high_gb": live_block * BYTES / GB,
        "params_and_grads_gb": PARAMS * (BYTES + 4) / GB,   # bf16 weights, fp32 grads
        "top_tensors": sorted(((v * BYTES / GB, k) for k, v in worst), reverse=True)[:6],
    }
    fixed = out["boundary_gb"] + out["block_recompute_gb"] + out["params_and_grads_gb"]
    out["total_gb"] = fixed + out["cotangent_low_gb"]
    out["total_high_gb"] = fixed + out["cotangent_high_gb"]
    return out


def largest_n(cap_gb, s=1, lo=64, hi=4096):
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if report(mid, s)["total_gb"] <= cap_gb:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[256, 512, 768, 1024, 1536])
    ap.add_argument("--card", type=float, nargs="+", default=[80, 141, 34.2],
                    help="GB: H100 SXM 80, H200 141, one Blackhole p150 34.2")
    ap.add_argument("--msa-rows", type=int, default=1)
    args = ap.parse_args()
    blob = {"per_n": [report(n, args.msa_rows) for n in args.n],
            "largest_n_that_fits": {str(c): largest_n(c, args.msa_rows) for c in args.card},
            "assumptions": "bf16, remat per block, 1 MSA row, XLA lower bound; see module docstring"}
    print(json.dumps(blob, indent=1))
    return blob


if __name__ == "__main__":
    main()
