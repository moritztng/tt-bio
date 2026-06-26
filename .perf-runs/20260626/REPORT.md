# tt-bio nightly perf — 2026-06-26

Branch: `exp/perf-20260626-triattn-transpose` (off `main` @ c892edb). No code shipped
(all attempts were measured dead-ends and reverted; tree clean). Card 0, tt-quietbox.

## TL;DR
Re-baselined Boltz-2 `--fast` on **current main** (the device-resident trunk recycling
loop has now MERGED to main — the old `baseline.md` predates it). Confirmed with fresh
per-op data that Boltz-2 `--fast` remains at its per-op compute ceiling. Two NEW measured
dead-ends recorded (ending-triattn transpose is intrinsic; trimul `reallocate` is
load-bearing — removing it triggers a fragmentation hang). No ≥5% validated win found;
the only remaining >5% lever is the trunk-only multi-device TP port (large/risky, no
reusable foundation on the old TP branch).

## New warm baseline — current main (post resident-trunk merge)
Warm = 2nd protein of a same-size pair, in-process program cache hot. Stage-synced.

| input        | total (s) | trunk | diffusion | confidence |
|--------------|-----------|-------|-----------|------------|
| 256 --fast   | 15.14     | 6.54  | **8.17**  | 0.43       |
| 512 --fast   | 29.46     | 18.43 | 9.61      | 1.42       |
| 686 --fast   | 51.21     | 33.50 | 14.64     | 3.07       |
| 512 default  | 40.70     | 28.58 | 10.09     | 2.03       |

(Old 2026-06-21 baseline.md had 512 --fast warm = 36.7s; the resident-trunk merge has
since brought it to 29.46s. baseline.md is stale — these supersede it.)

Notable: at L=256 **diffusion (8.17s) > trunk (6.54s)** — diffusion is the dominant stage
for small proteins. Diffusion barely scales with L (8.2 → 9.6 → 14.6), unlike the O(L³)
trunk.

## Per-op device-time breakdown (L=512 --fast, summed over both proteins ≈2×)
```
TriangleMultiplication = 13.13s   (n=1120)   <- #1, O(L³) cubic matmul
TriangleAttention      = 10.46s   (n=1120)   <- #2, SDPA over S×S
AttentionPairBias      =  7.68s   (n=12528)  <- mostly the diffusion DiT (200 steps), not trunk
Transition             =  4.97s   (n=1920)
```
Triangle ops (trimul+triattn) ≈ 56% of trunk device time — matches the documented ceiling.
Both ops are already heavily optimized (fused gp/qkv matmuls, `minimal_matmul`, SDPA
program configs, fp8 activations, sigmoid-fused gating). No surprise/regression to exploit.

## NEW finding 1 — ending-TriangleAttention transpose is intrinsic (+1.1s/protein @512)
The "ending" triangle attention runs the same kernel as "starting" on a transposed pair
tensor: it physically `permute`s the full `[S,S,c]` z twice per call (in + out;
tenstorrent.py:629/733, with an existing "CACHE -> RESHAPE PROBLEM" code comment).

Measured ending vs starting TriangleAttention (L=512 --fast, both proteins):
`end=6.32s  start=4.12s` → ending is **53% slower**, ≈1.1s/protein of pure transpose
overhead (≈3.7% of e2e @512, scales with L²).

Tested hypothesis that the post-permute tensor is non-contiguous (slow downstream path):
adding `ttnn.reallocate` after the ending permute → **no change** (end stayed 6.32s). So
the cost is the genuine TILE-layout retile of swapping the two sequence axes, not a
downstream layout issue. Eliminating it requires a custom "attend-over-batch-axis" kernel
(transposing q/k/v heads instead is strictly larger data). **Not worth it / intrinsic.**

## NEW finding 2 — the triangle permute path is CACHE-FRAGILE (two ways to break it)
Two independent edits to the triangle ops' permutes BOTH hung the **warm** (2nd, program-
cache-hot) protein at "trunk 0/4" while the cold protein completed fine — worker spinning
at 116% host CPU, 12+ min (vs ~18s normal), ignoring SIGINT/SIGTERM (needed SIGKILL):

- **(a) Removing `reallocate` in `TriangleMultiplication._transform_chunk`** (env-gated):
  the per-chunk concat loop fragments DRAM and the allocator thrashes. The reallocate is a
  deliberate anti-fragmentation guard, NOT overhead.
- **(b) Swapping `ttnn.permute(x,(1,0,2))` → `ttnn.transpose(x,0,1)`** for the ending-
  triattn axis swap: same warm hang. This is exactly what the "CACHE -> RESHAPE PROBLEM"
  code comment warns about — altering the triangle permute produces a program-cache /
  memory-layout state that deadlocks on warm reuse.

Both reverted (tree clean). Lesson: **do not touch the triangle-op permutes** — they are
load-bearing for the warm program-cache path, and breakage manifests only on the 2nd
(warm) protein, not the cold one. A SIGKILL'd hung worker left a chip dirty (orphaned
PPID=1 spawn worker holding the device; note `--device_ids 1` → `/dev/tenstorrent/2`, the
logical-vs-node mismatch); recovered with `tt-smi -r 0,1,2,3` (all cards verified healthy
afterward, warm p256b 15.0s).

## Diffusion is DiT-compute-bound even at L=256 (host glue only 9%)
Instrumented the 200-step sampler at L=256 (warm): `total=7.01s  dev_pnf=6.39s (91%)
kabsch=0.12s  host_other=0.49s`. The score-model device call dominates; a device-resident
eager coord-loop (removing host augmentation/Kabsch round-trips) could reclaim at most the
0.49s host_other ≈ <4% e2e. The DiT-internal host dispatch at small L is the real cost —
addressed only by the diffusion trace (already branched, default-off, gated >384 due to the
chaotic-fold concern). Conditioning is already fully hoisted (`_s_conditioned`/`_c_reshaped`
cached; only per-step `fourier(times)` recomputed, which is genuinely step-dependent).

## Multi-device TP trunk (the one >5% lever) — not tractable overnight
The 2026-06-24 op-level win (channel-shard trimul, 1.44–2.42× bit-identical) is real, but
the e2e path regressed and the resumable fix (trunk-only mesh + keep z channel-sharded
across all pairformer ops + amortize the per-call all_gather) is a from-scratch Pairformer
re-architecture. The old TP branch is based on a stale main and carries no reusable mesh
foundation. High risk of a broken/OOM tree by morning — deliberately NOT attempted unattended.

## Recommendation
No merge tonight. Boltz-2 `--fast` is at its per-op compute ceiling on current main; every
knob (resident trunk [merged], trace [regresses trunk / gated diffusion], bf8/HiFi2,
SDPA/subblock tuning, reallocate removal) is exhausted or harmful. The single remaining
>5% opportunity is the **trunk-only multi-device TP** — a scoped multi-day port, not an
overnight knob. Recommend it be tackled as a focused, supervised project, not a nightly run.
