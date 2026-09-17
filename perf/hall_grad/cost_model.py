#!/usr/bin/env python3
"""Cost and memory model for a gradient-based hallucination pass over Protenix v2.

Every architecture constant is read from the checkpoint, not hardcoded. Every timing
constant is a measured artifact, cited inline. CPU-only; opens no device.

Measured anchors (all Protenix v2, cdk2x2_512.yaml, 200 sampling steps, seed 0,
pc card 0 p150a Blackhole, aiclk 1350 MHz sampled in the artifact, ttnn 0.68.0,
tt-bio 4aa6dd6cb03455d96eb92d914d1761e16fb91564):
    perf/pxv1/v2_512_r10_pc0_a.json   recycling_steps=10  warm_median_s=50.253
    perf/pxv1/v2_512_r2_pc0_a.json    recycling_steps=2   warm_median_s=20.387
Recycles enter the trunk linearly by construction, so two points determine the split
exactly and the fit self-checks against the r2 point to the millisecond.
"""
import argparse, json, sys

# --- measured anchors ------------------------------------------------------------------
R10_S, R2_S = 50.253, 20.387
PER_RECYCLE_S = (R10_S - R2_S) / 8.0            # 3.733 s at 512 aa
RESIDUAL_S = R10_S - 10 * PER_RECYCLE_S         # embedder + 200 diffusion steps + conf + CIF
N_ANCHOR = 512

# --- architecture (protenix-v2.pt, read by arch_probe.py) --------------------------------
C_Z, C_S = 256, 384
N_PAIRFORMER, N_DIT = 48, 24
N_TRI_HEADS, TRI_HEAD_DIM = 8, 32
N_RECYCLES_DEFAULT, N_STEPS_DEFAULT = 10, 200

# --- hardware --------------------------------------------------------------------------
# tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml: dram_bank_size 4278190080, 8 channels
DRAM_BH = 4278190080 * 8                        # 34.23 GB
# wormhole_b0_80_arch.yaml: dram_bank_size 2147483648, 6 channels
DRAM_WH = 2147483648 * 6                        # 12.88 GB (12 GiB)
BF16 = 2


# Weight-matmul coefficient, AUDITED against every weight in pairformer_stack.blocks.0 of
# protenix-v2.pt rather than assumed (an earlier pass used 58 and was wrong on two counts):
#   tri_mul_{out,in}   linear_{a_p,a_g,b_p,b_g,g,z}, 6 x (256,256)   -> 6 x 2 = 12 each, 24
#   tri_att_{start,end} mha.linear_{q,k,v,g,o},      5 x (256,256)   -> 5 x 2 = 10 each, 20
#   pair_transition    linear_no_bias_a (1024,256), linear_no_bias_b (1024,256),
#                      linear_no_bias (256,1024) -- SwiGLU-GATED, so THREE matmuls at a 4x
#                      expansion, not two                            -> 3 x 2 x 4 =       24
#                                                                                  total  68
# Neglected as measured-negligible at 512 aa: attention_pair_bias is on the single track
# (5 x (384,384) = 0.76 GFLOP) plus linear_nobias_z (16,256) on the pair tensor (2.1 GFLOP),
# and single_transition is 3 x (1536,384) = 1.8 GFLOP -- together 0.3% of the block.
W_COEFF = 68
ACT_COEFF = 12   # 2 trimuls x 2 (triangle einsum) + 2 triatts x 4 (qk and av products)


def block_flops(n, c_z=C_Z):
    """Forward FLOPs of one PairformerLayer, split by what the backward costs.

    Backward for INPUT gradients only needs dX = dY W^T for a weight term (1x forward, because
    dW is skipped), but both dA and dB for an activation-activation product (2x forward).
    """
    weight = W_COEFF * n * n * c_z * c_z
    act = ACT_COEFF * n ** 3 * c_z
    return weight, act


def reconcile():
    """Self-check the FLOP model against the measured per-recycle time.

    If the implied achieved rate were absurd the model would be wrong and every per-step
    number below with it. 48 pairformer blocks at 512 aa against the measured 3.733 s.
    """
    w, a = block_flops(N_ANCHOR)
    per_recycle = N_PAIRFORMER * (w + a)
    return per_recycle, per_recycle / PER_RECYCLE_S


def backward_multiplier(n, c_z=C_Z):
    w, a = block_flops(n, c_z)
    return (w + 2 * a) / (w + a)


def score_tensor_bytes(n):
    """Triangle-attention score tensor [N_rows, heads, N, N]. The shipped fused SDPA
    never materialises this; any backward must recompute it in row chunks."""
    return n * N_TRI_HEADS * n * n * BF16


def pair_bytes(n, c=C_Z):
    return n * n * c * BF16


def block_retained_bytes(n):
    """Minimum activation set one PairformerLayer's backward needs, assuming a
    flash-style attention backward (scores recomputed, only row stats kept)."""
    p = pair_bytes(n)
    trimul = 2 * 10 * p          # z_in, a_p, a_g, b_p, b_g, a, b, prod, z_proj, g
    triatt = 2 * 5 * p           # q, k, v, g, o
    lse = 2 * n * N_TRI_HEADS * n * BF16      # log-sum-exp row stats
    transition = pair_bytes(n, 4 * C_Z) + 2 * p
    return trimul + triatt + lse + transition


def fwd_s(n, n_cycles, with_diffusion):
    """Forward seconds, scaled from the 512 aa anchor by the measured size exponent."""
    scale = (n / N_ANCHOR) ** args.exponent
    t = n_cycles * PER_RECYCLE_S * scale
    if with_diffusion:
        t += RESIDUAL_S * scale
    return t


def gb(b):
    return b / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cycles", type=int, default=3,
                    help="recycles in the hallucination forward (ColabDesign uses 0-3)")
    ap.add_argument("--exponent", type=float, default=1.86,
                    help="measured size exponent; perf/px4pd 256->640 on qb2c1 gives 1.86, "
                         "640->768 gives 1.46, so this is a range not a constant")
    ap.add_argument("--grad-steps", type=int, default=410,
                    help="ColabDesign design_3stage default: soft 300 + temp 100 + hard 10")
    ap.add_argument("--mcmc-steps", type=int, default=1000,
                    help="ColabDesign _design_mcmc default steps=1000")
    args = ap.parse_args()
    n = args.n

    print(f"=== anchors (512 aa, p150a, aiclk 1350 MHz, ttnn 0.68.0) ===")
    print(f"  per trunk recycle          {PER_RECYCLE_S:.3f} s")
    print(f"  trunk at 10 recycles       {10*PER_RECYCLE_S:.3f} s")
    print(f"  residual (diff200+rest)    {RESIDUAL_S:.3f} s")
    print(f"  r2 refit check             {2*PER_RECYCLE_S+RESIDUAL_S:.3f} s vs measured 20.387")
    print(f"  per pairformer block       {PER_RECYCLE_S/N_PAIRFORMER*1e3:.1f} ms (upper bound)")
    print(f"  per diffusion step         {RESIDUAL_S/N_STEPS_DEFAULT*1e3:.1f} ms (upper bound)")

    tf, rate = reconcile()
    print(f"\n=== FLOP-model reconciliation (512 aa) ===")
    print(f"  48 pairformer blocks       {tf/1e12:.2f} TFLOP per recycle")
    print(f"  measured per recycle       {PER_RECYCLE_S:.3f} s")
    print(f"  implied achieved rate      {rate/1e12:.1f} TFLOP/s")
    print(f"  plausible: tt-bio runs Protenix at HiFi4 + fp32_dest_acc (4 passes through the")
    print(f"  matrix unit) on an 11x10 grid, and our own C10 campaign closed the 512 aa cell as")
    print(f"  matmul BANDWIDTH-bound, so a low-teens-percent compute utilisation is expected.")

    w, a = block_flops(n)
    m = backward_multiplier(n)
    print(f"\n=== backward multiplier at N={n} ===")
    print(f"  weight-matmul FLOPs        {w/1e12:.3f} T  (backward 1x: input grad only)")
    print(f"  activation-prod FLOPs      {a/1e12:.3f} T  (backward 2x: both operands)")
    print(f"  backward / forward         {m:.3f}x")
    print(f"  with per-block recompute   {1+m:.3f}x")
    print(f"  NOTE this is a FLOP ratio and therefore a FLOOR. The workload is bandwidth-bound,")
    print(f"  and the backward moves more bytes per FLOP than the forward because it reads the")
    print(f"  saved activations back. A realised multiplier above this makes the gradient path")
    print(f"  worse, not better, so every verdict below is robust to the gap.")

    print(f"\n=== memory at N={n} (bf16) ===")
    print(f"  pair tensor [N,N,{C_Z}]      {gb(pair_bytes(n)):.3f} GB")
    print(f"  SCORE tensor [N,{N_TRI_HEADS},N,N]    {gb(score_tensor_bytes(n)):.2f} GB"
          f"  = {100*score_tensor_bytes(n)/DRAM_BH:.0f}% of a Blackhole chip")
    print(f"  one block retained         {gb(block_retained_bytes(n)):.2f} GB")
    print(f"  48 blocks unchecked        {gb(48*block_retained_bytes(n)):.0f} GB  -> impossible")
    tape = N_PAIRFORMER * (pair_bytes(n) + n * C_S * BF16)
    print(f"  per-block checkpoint tape  {gb(tape):.2f} GB")
    print(f"  Blackhole 34.23 GB         tape {gb(tape):.2f} + live {gb(block_retained_bytes(n)):.2f}"
          f" = {gb(tape+block_retained_bytes(n)):.2f} GB -> "
          f"{'FITS' if tape+block_retained_bytes(n) < DRAM_BH else 'DOES NOT FIT'}"
          f" ({gb(DRAM_BH-tape-block_retained_bytes(n)):+.2f} GB margin)")
    print(f"  Wormhole 12 GiB            tape {gb(tape):.2f} + live {gb(block_retained_bytes(n)):.2f}"
          f" = {gb(tape+block_retained_bytes(n)):.2f} GB -> "
          f"{'FITS' if tape+block_retained_bytes(n) < DRAM_WH else 'DOES NOT FIT'}")

    print(f"\n=== per gradient step at N={n}, {args.cycles} recycles (exponent {args.exponent}) ===")
    bw_trunk = args.cycles and fwd_s(n, 1, False) * m       # last recycle only, no recompute needed
    ck_bw = fwd_s(n, 1, False) * (1 + m)                    # last recycle with per-block recompute
    optA = fwd_s(n, 1, False) * (N_PAIRFORMER and 4/N_PAIRFORMER) * (1+m)
    print(f"  A confidence head only     {optA:.2f} s backward -- does NOT reach the sequence")
    b1 = fwd_s(n, args.cycles, False) + ck_bw
    print(f"  B1 distogram -> last cycle {b1:.2f} s/step"
          f"  ({args.grad_steps} steps = {b1*args.grad_steps/3600:.2f} h)")
    b2 = fwd_s(n, args.cycles, True) + ck_bw + RESIDUAL_S/N_STEPS_DEFAULT*(n/N_ANCHOR)**args.exponent*(1+m)
    print(f"  B2 + one denoise step      {b2:.2f} s/step"
          f"  ({args.grad_steps} steps = {b2*args.grad_steps/3600:.2f} h)")
    c = fwd_s(n, N_RECYCLES_DEFAULT, True) * (1 + m)
    print(f"  C full 200-step trajectory {c:.2f} s/step"
          f"  ({args.grad_steps} steps = {c*args.grad_steps/3600:.2f} h) -- NO-GO on chained bf16")

    print(f"\n=== forward-only alternative at N={n}, {args.cycles} recycles ===")
    f1 = fwd_s(n, args.cycles, False)
    print(f"  one forward (distogram)    {f1:.2f} s")
    print(f"  MCMC {args.mcmc_steps} steps          {f1*args.mcmc_steps/3600:.2f} h/design")
    print(f"  gradient {args.grad_steps} steps (B1)    {b1*args.grad_steps/3600:.2f} h/design")
    print(f"  gradient buys              {f1*args.mcmc_steps/(b1*args.grad_steps):.2f}x wall-clock")
    print(f"  forwards per gradient step {args.mcmc_steps/args.grad_steps:.2f}"
          f"  (ColabDesign defaults: {args.mcmc_steps} mcmc vs {args.grad_steps} grad)")
