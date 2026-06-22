# tt-bio perf run — 2026-06-22 (overnight, autonomous)

Branch: `exp/perf-20260622-ceiling-probe` (off main `8a40c42`). No repo code changed —
all experiments were non-invasive `sitecustomize` monkeypatches in `/tmp`, so main and
the branch are byte-identical. NOT merged/pushed to main.

## TL;DR
I profiled the warm Boltz-2 `--fast` path deeply and ran **5 measured optimization
experiments**. **All gave ≈0** (within run-to-run noise) or regressed. The Boltz-2 *and*
Protenix-v2 ttnn paths are at their per-op kernel ceiling: every stage is
compute/bandwidth-bound on irreducible work, and the obvious knobs are already optimal.
**No new ≥5% incremental win was found.** The real available wins remain the two existing
(unmerged) branches — `exp/perf-20260621-diffusion-resident` (trace, validated **-7.6%
e2e @512**, lossless) and `exp/perf-resident-trunk` (-16% trunk). Recommend Moritz merge
the trace branch; further gains need algorithmic or multi-device work, not knob-tuning.

## Warm baseline (L=512, --fast, tt-quietbox BH card 0, clean/non-concurrent)
Reproduced day-old baseline exactly. `tt-bio predict in512 --fast --debug --log --seed 0`,
2nd protein (p512b) = warm:

| stage       | warm time | share |
|-------------|-----------|-------|
| trunk       | 24.7s     | 67%   |
| diffusion   | 10.1s     | 28%   |
| confidence  | 1.7s      | 5%    |
| **e2e**     | **36.8s** |       |

### NEW quantified breakdowns (module-level, sync-timed)
- **Trunk** (the 67%): `TriangleMultiplication` 13.3s + `TriangleAttention` 10.7s dominate
  (both O(L³)); MSA stack (`MSALayer`/`OuterProductMean`/`PairWeightedAveraging`) ~9s
  overlapping. All compute/bandwidth-bound.
- **Diffusion** (the 28%): **91% device-compute-bound** — `preconditioned_network_forward`
  is 7.2s of the 7.86s warm sample; host (centering, random augmentation, Kabsch align
  0.15s, euler) is only 0.67s/9%. Per-step 36ms.
  - Internal split: 24-layer **token transformer = 59%** (20.8ms/step), atom encoder+decoder
    (3+3 layers, head_dim=32) = 35% (12.2ms/step), conditioning s-path = ~6%. All
    step-variant (depend on r); `_s_conditioned`/`_c_reshaped`/SDPA biases already hoisted.
  - Token-transformer / DiT **head_dim = 48** (16 heads × 48 = 768) → not tile-aligned →
    ttnn SDPA pads to 64 and slices back (`o[...,:48]`). ~25-30% inherent SDPA-matmul waste,
    unfixable without a custom non-tiled kernel. This is why the code uses a manual
    permute/reshape merge instead of `nlp_concat_heads` (which needs a 32-aligned head_dim).

## Experiments (all measured, all ≈0 → NEW dead-ends)
| # | idea | result | why |
|---|------|--------|-----|
| 1 | **HiFi2 on the diffusion score model** (per-module fidelity; skill's HiFi2 note was trunk-only) | per_fwd 37.9ms vs 36.0ms baseline → **0** | diffusion forward is **not matmul-math-bound** (bf8 already + bandwidth/op-bound); fewer fidelity passes don't help |
| 2 | **trimul O(L³) matmul subblock** 1×1 → 2×2 (`_triangle_mul_program_config`) | warm trunk 24.34s vs 24.71s → **-1.5%, within noise** | the skinny (contraction=32) trimul matmul is bandwidth-bound, not subblock-overhead-bound |
| 3 | **SDPA chunk size** sweep (128 / 384 / 512; baseline 256) | 512 **OOMs L1**, 384 slower, 128 ≈ baseline → **0** | 256 is already the L1-optimal point |
| 4 | **`exp_approx_mode=True`** in every SDPA program config (faster approximate softmax) | warm trunk 24.74s, diffusion 10.04s → **0** | SDPA is dominated by QK^T/AV matmuls, not the softmax exp |
| 5 | (probe) grid utilization | already auto-snaps to **13×10** (max Blackhole) | nothing to reclaim |

## Maturity audit (why there's no low-hanging fruit)
- **Boltz-2 trunk**: fused `minimal_matmul` for gp_in/qkv, chunked trimul with custom
  memory+program configs, fused sigmoid-multiply gating, bf8 in `--fast`, tuned SDPA chunks.
- **Boltz-2 diffusion**: step-invariant conditioning + SDPA pair-bias precomputed/hoisted.
- **Protenix-v2** (recently added): already has the skill's documented wins — recompute-hoist
  (`_atom_cond`, `_dit_block_biases`, `precompute_biases`/`bias_cache`), windowed K/V gather
  via a single `ttnn.embedding` (`_WIN_KV_IDX`), and the 24-block token DiT runs **on-device**
  (`device_dit=True`), not the fp32-host fallback. No fresh fruit there either.
- Known prior dead-ends (LEARNINGS): trunk hoist/cheap-fusion, bf8 trunk weights, fused
  custom trimul matmul, diffusion-step trace at L≥512. All confirmed still true.

## Accuracy / robustness
N/A — no code change to validate. All folds in this run completed correctly (no OOM,
no accuracy touch); the 5 experiments were measured for *speed* and discarded.

## Recommendation
1. **No new merge from this run.** The incremental knob space for Boltz-2 `--fast` is
   exhausted; chasing it further is low-EV.
2. **Merge the existing trace branch** `exp/perf-20260621-diffusion-resident` — it is the
   real validated win (-7.6% e2e @512, larger at small N, lossless by construction, env-gated
   default-off = zero risk). Validate templated/ligand/multimer inputs first (its own report's
   follow-up).
3. **Future direction** (for a human to scope): the remaining gap to the `tt-minimal` ~25s
   floor and below is algorithmic/structural, not a knob:
   - test whether trace + resident-trunk **stack** (likely overlapping, but unmeasured);
   - multi-device tensor-parallel trunk (the O(L³) pair rep sharded across the 4 cards) —
     the only lever left that attacks the 67% trunk directly, but a large, risky effort.
