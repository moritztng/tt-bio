# Token-DiT attention (AttentionPairBias) kernel scout

## Result

No production change is justified by this pass.

The token-level DiT attention half (`AttentionPairBias` inside the 24-block
`DiffusionTransformer`) is a large, real share of a Protenix-v2 fold, but it is
already collapsed to the minimum op stream: the pair-bias injection is precomputed
once per fold and replayed as the SDPA mask (zero per-step ops), and QKV is already
one packed linear. The single remaining avoidable dispatch (the attention gate
projection `proj_g`) can be folded into the packed QKV linear, but doing so is a
net **end-to-end regression** and is **not lossless** (5.2 Angstrom coordinate
drift), so it is rejected.

| Region | share |
|---|---:|
| token DiT / diffusion step | **61.7%** |
| AttentionPairBias / token DiT | 32-38% |
| **AttentionPairBias / diffusion step** | **~20%** |
| **AttentionPairBias / full fold** | **~10%** |
| attention half (apb + attn-side AdaLN) / diffusion step | ~30% |

* The share is comfortably above the ~5% disqualifying threshold, but there is no
  fusion headroom: SDPA and the two projections (packed QKV, output) are the
  necessary near-peak matmul/attention compute, and the two fusions a naive profile
  would suggest (pack QKV, fold in the pair bias) are **already in the shipping
  baseline**.
* The one residual pack (fold `proj_g` into the packed QKV linear, both left-multiply
  the same AdaLN'd input) runs at **0.97x** end-to-end and diverges coordinates by
  **5.2 A RMSD** at the production 200 steps. Same failure family as the
  TriangleAttention QKV-pack (0.58-0.61x) and the atom-attention K+V-pack (0.928x):
  micro-packing loses on this regime, and here it also cannot clear diffusion's
  bit-exact bar.
* As with the atom attention (docs/atomattention-kernel-scout.md), the real lever for
  this dispatch-bound per-step stream is the whole-denoise trace replay that already
  exists (`fold(trace=True)`). A per-op pack gives the traced path nothing (identical
  device kernels) and the untraced path a regression. Runtime code is unchanged, so no
  accuracy gate was required.

## Method

Measurements used qb1 physical card 3 (one Blackhole P150a), real Protenix-v2
checkpoint weights, and `examples/prot.yaml` (117 tokens). Every timed region is the
second same-shape forward in one process and is bracketed by a device sync, matching
the prior scouts (docs/permodel-kernel-scout.md, docs/atomattention-kernel-scout.md).

Two timed folds separate the coarse and fine shares so the inner per-call syncs never
inflate the coarse denominators:

* **coarse** wraps only `denoise` and `_token_dit_device` (2 syncs/step) ->
  token-DiT/denoise and denoise/fold.
* **fine** additionally wraps the 24 per-block `AttentionPairBias` and attention-side
  `AdaLN` instances -> apb/token-DiT within the same synced run.

Because the fine run's `_token_dit_device` time carries its own inner syncs, the true
apb share sits between the same-run ratio (fine token-DiT denominator) and the ratio
against the un-inflated coarse token-DiT: 32.1% and 38.3% respectively, i.e. ~20-24%
of the diffusion step.

## Share

`prot.yaml`, 1 sample, 200 sampling steps (production default), warm:

| Region | per step | share |
|---|---:|---:|
| full warm fold | - | 11.10 s total |
| diffusion denoise | 29.0 ms | 52.3% of fold |
| token DiT (24 blocks) | 17.9 ms | 61.7% of denoise |
| AttentionPairBias (x24) | ~0.28 ms/call | 32.1% of token DiT |
| attention-side AdaLN (x24) | - | 16.7% of token DiT |
| attention half (apb + AdaLN) | - | 48.8% of token DiT |

token DiT is 61.7% of a Protenix diffusion step, matching the 67.1% measured for a
Boltz-2 diffusion step in the prior atom-attention scout. The attention projections +
SDPA + output (apb) are ~20% of the step and ~10% of the whole fold; adding the
attention-side AdaLN brings the attention half to ~30% of the step.

## Dispatch decomposition

One warm token-level apb call (`bias_precomputed=True`, the per-step DiT case) issues
17 ttnn launches:

| op | launches | op | launches |
|---|---:|---|---:|
| deallocate | 6 | linear | 3 |
| permute | 2 | multiply | 1 |
| reshape | 1 | unsqueeze | 1 |
| slice | 1 | nlp_create_qkv_heads | 1 |
| scaled_dot_product_attention | 1 | | |

The 3 linears are the packed QKV projection, the gate `proj_g`, and the output
projection. There is **no per-step pair-bias op**: the `layer_norm(z) + linear +
permute` that derives the per-head bias is a pure function of the fixed trunk pair_z,
so it is precomputed once per fold (`_dit_block_biases`) and passed straight into SDPA
as `attn_mask`. QKV is already a single packed linear with head-padded weights. The
two fusions a fresh profile would reach for are therefore already shipping; the only
remaining adjacent pair is QKV + gate (both consume the same input).

## Gate-pack A/B

Fold `proj_g` into the packed QKV linear: `[Wqkv | Wg]` as one matmul on the shared
input, then slice off `g` (tile-aligned, cheap) for the post-SDPA gate. Removes one
linear launch.

Isolated warm microbenchmark (200 reps, real captured DiT input, deallocates matched):

| | median | parity |
|---|---:|---:|
| baseline (separate QKV, g) | ~0.28 ms | reference |
| packed QKV+gate | ~0.25 ms | PCC 1.0000, max-abs 0.125 (rel 0.24%) |
| isolated speedup | ~1.05-1.15x (noisy) | not bit-exact |

The isolated call looks slightly faster because the warm loop is host-dispatch bound,
where one fewer launch wins. But end-to-end it inverts. Full-fold A/B (same seed,
device-synced wall clock; baseline reproduces itself at 0.0 A RMSD, confirming
bit-determinism):

| steps | e2e speedup | coord RMSD | coord max-dev |
|---:|---:|---:|---:|
| 20 | 0.994x | 2.92 A | 10.0 A |
| 200 (production) | 0.971-0.982x | **5.21 A** | 12.3 A |

Two independent disqualifiers:

1. **Regression.** In the real fold the device is the bottleneck, and the 3840-wide
   packed matmul is marginally slower than the separate 3072 + 768 matmuls; the saved
   launch does not pay for it. Net 0.97x.
2. **Not lossless.** The pack changes the matmul tiling, so the output differs by a
   bf16 half-ULP (max-abs 0.125 at magnitude 52.5). Per call that is PCC 1.0, but the
   diffusion sampler is chaotic and bit-exact-sensitive: the per-op drift compounds
   over 24 blocks x 200 steps to 5.2 A of coordinate change. RMSD grows with step
   count (2.9 A at 20 -> 5.2 A at 200), the signature of trajectory compounding rather
   than a systematic error.

## Reproduce

```bash
cd ~/.coworker/wt/tt-bio-boltz2-dit-attention-kernel-scout   # or any tt-bio checkout
# coarse shares (denoise/fold, token-DiT/denoise)
PYTHONPATH=. TT_VISIBLE_DEVICES=3 python3 scripts/dit_attention_kernel_scout.py share --steps 200
# fine shares (apb/token-DiT, attn-side AdaLN)
PYTHONPATH=. TT_VISIBLE_DEVICES=3 python3 scripts/dit_attention_kernel_scout.py attn --steps 200
# per-call ttnn op-launch decomposition
PYTHONPATH=. TT_VISIBLE_DEVICES=3 python3 scripts/dit_attention_kernel_scout.py decomp
# gate-pack A/B: isolated microbench (parity + warm timing)
PYTHONPATH=. TT_VISIBLE_DEVICES=3 python3 scripts/dit_attention_kernel_scout.py ab --repeats 200
# gate-pack A/B: full-fold e2e (wall clock + coordinate RMSD, same seed)
PYTHONPATH=. TT_VISIBLE_DEVICES=3 python3 scripts/dit_attention_kernel_scout.py e2e --steps 200
```
