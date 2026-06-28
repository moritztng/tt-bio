# 2026-06-28 — Boltz-2 trunk MSA module: characterized (NEW) + PWA head-batching dead-end

Branch: `exp/perf-20260628-msa-trunk` (off main `c892edb`). **No code shipped** — the one
attempt (PWA head-batching) is a measured ~0.3% e2e dead-end, reverted. Tree clean.

## Angle this time: the trunk MSA module (the one trunk component 6 nights never broke out)
Every prior night dissected the Pairformer triangle ops (trimul/triattn). The trunk also
runs an **MSA module** (`MSAModule`, `n_blocks=4`) each recycle, and its share was never
measured. I profiled it with a device-synced hook (`/tmp/msahook`, gated `TT_MSA_PROFILE=1`,
edits no repo files).

## NEW measurement — warm trunk internal breakdown @512 (device-synced, steady-state warm)
| component            | warm/call | calls/protein | warm/protein | share of e2e |
|----------------------|-----------|---------------|--------------|--------------|
| Pairformer module    | 2.62s     | 5 (4 recycle + glue) | ~13.1s | ~43% |
| **MSA module**       | **0.95s** | 4             | **~3.8s**    | **~12.5%**   |
| └ PairWeightedAveraging | 0.089s | 16 (4/recycle)| ~0.36s       | ~1.2%        |
| └ OuterProductMean   | ~0.06s    | 16            | ~0.24s       | ~0.8%        |
| └ (rest: msa_transition + embedded pairformer_layer trimul/triattn) | | | ~3.2s | at-ceiling |

So the MSA module is a real ~12.5% of e2e — but **most of it is its embedded
`pairformer_layer`** (one trimul + one triattn on z per MSA block), which is the SAME
at-ceiling primitive the last 6 nights settled. The **MSA-specific** ops (PWA + OPM +
msa_transition) are only ~0.7–1.0s/protein combined.

MSA tensor shapes (`--fast`): m = `[1, N_msa, S, 64]` with **N_msa padded to 4096**
(256: 4058→4096, 512: 3627→4096, 686: 2566→3072). z = `[1, S, S, 128]`. PWA: 8 heads,
head_dim 32. `MSA_PAD_MULTIPLE = 1024` (tenstorrent.py:51) — a deliberate bucketing choice
so different proteins share an N_msa bucket and hit the warm program cache; reducing it would
break warm caching across proteins (the documented ttnn-bucketing trade-off). Do not touch.

## Attempt (reverted): batch the PairWeightedAveraging per-head loop
PWA loops `for i in range(8)` launching ~88 small ops/call (per-head b-projection, permute,
mask-add, softmax, v/g projections, matmul, gate, out-projection + accumulate). I rewrote it
to project the per-head attention bias in ONE matmul + ONE softmax over all heads
(bit-identical), keep the value-average/gate streamed per head (m spans the full padded MSA
depth — a fully batched value tensor OOMs at N_msa≥4096), and replace the 8 per-head
out-projections + 7 adds with one `concat(heads) @ full o_weight`.

- **Parity:** `test_tenstorrent.py::test_msa` (seq_len 100/500/1000, real weights) passes
  before and after (median rel-err < 0.1).
- **Warm A/B @512 (device-synced, steady-state):** PWA 0.0891s → **0.0817s/call (−8.3%)**,
  consistent across every warm call (variance < 0.5%). MSA module 0.953s → 0.928s/call.
- **e2e impact:** PWA is only ~0.36s/protein, so −8.3% on it = **~0.1s/protein ≈ 0.3% of e2e**.
  Warm trunk 18.72s → 18.68s (within run-to-run noise). **Below the 5% bar.**

### Why it's near-zero (the lesson)
My initial estimate (88 ops × ~1ms = launch-bound, batchable) conflated **cold-compile**
calls with **warm dispatch**. The `device-synced sum over cold+warm` (0.097s avg) over-stated
the warm cost; the true warm PWA is 0.089s/call and is dominated by the **8 per-head matmuls**
(`v[N,32,S] @ w[S,S]`, the real FLOP), not op-launch overhead. On the warm program-cache path
the per-head dispatch is cheap, so batching reclaims only the ~8% that was launch overhead.
The MSA-specific ops are FLOP/bandwidth-bound on the (deliberately bucketed) N_msa=4096 depth
— at-ceiling, like the Pairformer and diffusion.

### Why not shipped
0.3% e2e, costs +~1GB transient DRAM (the all-heads out-proj concat), and the out-proj
reorder is no longer bit-identical (PCC>0.99, not maxdiff=0). Not worth the memory/validation
cost for a sub-noise gain. Reverted; tree clean.

## Verdict / recommendation
- **MSA module is at-ceiling** (NEW): ~12.5% of e2e, but dominated by its embedded
  trimul/triattn (already settled); the MSA-specific PWA/OPM are ≤1.2% / ≤0.8% of e2e each and
  FLOP-bound on a bucketing-locked N_msa. No op-level lever ≥5% here. **Abandoned.**
- A possible (untaken) PWA-internal micro-win: the per-head value matmul
  `v[N,hd,S] @ w[S,S]` is a skinny **batched** matmul (32-row LHS); because w is shared across
  the N_msa batch, it could be reformulated as one **fat 2D** `[N*hd, S] @ [S,S]` matmul. Capped
  at ≤0.5% e2e (PWA is only 1.2%) and reshape-fragile — noted, not pursued.
- **No ≥5% lever remains in Boltz-2 `--fast` op kernels** (confirmed across 7 nights). The only
  path to a ≥5% e2e win is the algorithmic **trunk-only multi-device TP** (the documented
  resumable path: keep z channel-sharded across trimul ops, diffusion stays single-device).
  That is a multi-day port, not an overnight knob — recommend a focused (attended) effort.

Baseline (warm, tt-quietbox card 0, `--fast`): 512f = 30.58s (tr 18.72 / df 10.16 / cf 1.70).
