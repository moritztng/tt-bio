#!/usr/bin/env python3
"""How many bytes a BioIR pairformer block moves, counted from the package source.

The one quantity the roofline write-up could not measure is arm C's DRAM traffic, and the brief
names the right free source for it: bionemo-ir 0.1.0 is on PyPI and its torch layer is readable.
So this counts the same thing our device capture measured -- bytes at pair-tensor scale crossing
DRAM in one pairformer block -- but from their code rather than from a profiler.

Unit is Z, one bf16 pair tensor: N x N x c_z x 2 bytes = 67.11 MB at N=512, c_z=128.

Every step below is a line in bionemo_ir/_torch/layers/{triangle_nodes.py, attention.py,
transformers/pairformer.py}. A fused custom op counts as one region: its inputs read once, its
outputs written once. That is the same accounting rule the ttnn capture applies to our side, so
the two numbers are comparable.

Boltz-2 runs this path: models/boltz2/config.py sets dtype="bfloat16" and
trimul_high_precision=False, so the .to(self.high_precision_dtype) casts in _forward_impl_v2 are
no-ops and nothing widens to fp32; and _forward_impl_v2_threshold = 384, so at 512 tokens the v2
path runs, not v1.

Triangle attention comes off TriangleAttention.forward the same way: one fused qkv_proj emitting
q|k|v concatenated, the flash kernel, then _gated_sigmoid_op folding the g projection and the
sigmoid gate into an in-place write over the attention output, then o_proj. Nothing below is
guessed.

Cross-check that the model is the right shape: it predicts six layer_norm_transpose calls per
pairformer block (two per trimul, one per triangle attention) plus three per diffusion token
layer, so 6 x 264 + 3 x 4800 = 15984 per fold. The census counts 16005.
"""
import json

N, C_Z = 512, 128
Z = N * N * C_Z * 2          # 67.11 MB, one bf16 pair tensor

TRIMUL = [
    ("layer_norm_transpose(x)", 1, 1),
    ("dual_gemm_x_x -> a,b (fused LN, gate, both projections)", 1, 2),
    ("einsum a,b -> x", 2, 1),
    ("layer_norm_transpose(x) out", 1, 1),
    ("dual_gemm_x0_x1 (fused out gate + projection)", 2, 1),
]
TRIATT = [
    ("layer_norm(x)", 1, 1),
    ("ln_proj_moveaxis_pad -> triangle bias", 1, 0),   # bias is 4/128 of Z, dropped
    ("qkv_proj, one fused Linear -> q|k|v", 1, 3),
    ("flash triangle attention -> o", 3, 1),           # + an lse companion, 0.06Z, dropped
    ("gated_sigmoid: g projection + gate, in place on o", 2, 1),
    ("o_proj", 1, 1),
]
TRANSITION_Z = [
    ("fused swiglu, 128 -> 512", 1, 4),
    ("output projection, 512 -> 128", 4, 1),
]
RESIDUALS = 5                                          # z = z + f(z), five times per block

# --- diffusion token layer, same treatment ---------------------------------------------
# transformers/diffusion_transformer.py::DiffusionTransformerLayer.forward, 512 tokens, dim 768.
A = 512 * 768 * 2                     # 0.786 MB, one token tensor
DIT_WEIGHTS = 11802624 * 2            # 23.6 MB, the layer's own parameters, bf16
DIT_BIAS = 512 * 512 * 16 * 2         # 8.39 MB, this layer's slice of the shared pair bias
DIT_ACT_A = (
    3      # adaln(a, s) -> b
    + 8    # pair_bias_attn: read b, write q|k|v, read them, write o
    + 3    # gated_sigmoid: g projection + gate over o
    + 3    # a = a + b
    + 9    # transition: adaln, swiglu to 1536, projection back
    + 3    # a = a + transition
)

# Our side, measured: fold_bytes_512.json.
OURS_BLOCK_BYTES = 12169659392        # pairformer block
OURS_DIT_BYTES = 214449152            # diffusion token layer
OUR_FOLD_TB = 6.23
OUR_PF_TB, OUR_MSA_TB, OUR_DIT_TB, OUR_ATOM_TB = 3.213, 0.189, 1.029, 0.620
OUR_REST_TB = OUR_FOLD_TB - (OUR_PF_TB + OUR_MSA_TB + OUR_DIT_TB + OUR_ATOM_TB)
THEIR_DEVICE_S, H200_ROOF_TBS = 2.0695, 4.8


def total(steps):
    return sum(r + w for _, r, w in steps)


def main():
    tri_mul, tri_att, trans = total(TRIMUL), total(TRIATT), total(TRANSITION_Z)
    block = 2 * tri_mul + 2 * tri_att + trans + 3 * RESIDUALS
    ours = OURS_BLOCK_BYTES / Z
    ratio = ours / block

    # Diffusion token layer, counted the same way.
    dit = DIT_WEIGHTS + DIT_BIAS + DIT_ACT_A * A
    dit_ratio = OURS_DIT_BYTES / dit
    their_pf_TB = OUR_PF_TB / ratio
    their_msa_TB = OUR_MSA_TB / ratio           # same block shape, same fusions
    their_dit_TB = 4800 * dit / 1e12

    # Two ends for what is still not counted: the atom transformer and the 1.18 TB our fold
    # spends outside the instrumented phases. Low end scales both by the pairformer ratio, high
    # end charges BioIR exactly what those cost us.
    low = their_pf_TB + their_msa_TB + their_dit_TB + (OUR_ATOM_TB + OUR_REST_TB) / ratio
    high = their_pf_TB + their_msa_TB + their_dit_TB + OUR_ATOM_TB + OUR_REST_TB
    pct = lambda tb: round(100 * tb / THEIR_DEVICE_S / H200_ROOF_TBS)  # noqa: E731

    out = {
        "Z_bytes": Z, "N": N, "c_z": C_Z,
        "bioir_tri_mul_Z": tri_mul, "bioir_tri_att_Z": tri_att,
        "bioir_transition_z_Z": trans, "bioir_residual_adds_Z": 3 * RESIDUALS,
        "bioir_block_Z": block, "bioir_block_GB": round(block * Z / 1e9, 3),
        "ttbio_block_Z": round(ours, 1), "ttbio_block_GB": round(OURS_BLOCK_BYTES / 1e9, 3),
        "ratio_ttbio_over_bioir": round(ratio, 2),
        "bioir_dit_layer_MB": round(dit / 1e6, 2), "ttbio_dit_layer_MB": round(OURS_DIT_BYTES / 1e6, 2),
        "ratio_ttbio_over_bioir_dit": round(dit_ratio, 2),
        "their_fold_TB": [round(low, 2), round(high, 2)],
        "their_pct_of_h200_bandwidth_roof": [pct(low), pct(high)],
        "our_pct_of_p150a_bandwidth_roof": 61,
    }
    print(json.dumps(out, indent=1))
    for name, steps in (("tri_mul", TRIMUL), ("tri_att", TRIATT), ("transition_z", TRANSITION_Z)):
        print("\n%s:" % name)
        for label, r, w in steps:
            print("  %-52s read %dZ write %dZ" % (label, r, w))


if __name__ == "__main__":
    main()
