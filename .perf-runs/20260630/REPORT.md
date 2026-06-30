# 2026-06-30 — Trunk tensor-parallel feasibility: head-shard triangle-attention (the untested lever)

Branch: `exp/perf-20260630-trunk-profile`  (standalone spike only; NO live-model edits)
Spike: `.perf-runs/20260630/mesh_trunk_spike.py`

## Why this experiment
Boltz-2 `--fast` is at the single-device compute ceiling: trunk trimul/triattn are
compute/bandwidth-bound and diffusion DiT is compute-bound even at L=256 (LEARNINGS).
Every single-device knob is a measured dead end. The ONLY documented remaining lever
is multi-device tensor parallelism on the trunk. The 2026-06-24 work sharded **only**
TriangleMultiplication (channel-TP, 1.75x@512, bit-identical) and found the whole-model
mesh REGRESSED e2e — partly because every *other* (replicated) op pays a mesh-dispatch
tax. The untested question: **does also sharding the 2nd dominant op-type — triangle
ATTENTION — tip the trunk into a net win?** Boltz-2 trunk triattn has exactly 4 heads
→ one head per Blackhole card, a clean head-parallel split. trimul(31.5%)+triattn(25%)
= 56.5% of trunk device time.

## Method
Standalone spike on current main, production config (Blackhole HiFi4, `--fast` bf8),
real `TriangleMultiplication` / `TriangleAttention` / `Transition` primitives, random
weights. Warm per-call time (program cache on, REPS=20, device-synced). Mesh = 1×4
RING. Validate sharded-vs-single PCC/maxdiff.

## Results (warm per-call, ms)

| L   | op         | single | mesh(1×4) | speedup | PCC vs single |
|-----|------------|--------|-----------|---------|---------------|
| 512 | trimul     | 9.69   | 5.62      | **1.72x** | maxdiff=0 (bit-identical) |
| 512 | triattn    | 6.46   | 5.19      | **1.25x** | maxdiff=2.4e-3 (PCC≈1) |
| 512 | transition | 4.88   | 7.52      | **0.65x** | (replicated — MESH TAX) |

- trimul 1.72x reproduces the 2026-06-24 channel-TP win, bit-identical (maxdiff=0).
- triattn head-shard (NEW): 1.25x @512, numerically equivalent (maxdiff 2.4e-3 from a
  benign head-split reduction-order change; not bit-identical but PCC≈1).
- **Replicated Transition is 1.54× SLOWER on the mesh (0.65x)** — direct measurement of
  the mesh-dispatch tax the 06-24 work blamed for the whole-model regression.
- Projected per-layer z-path (2×trimul + 2×triattn + transition): single 37.2ms →
  mesh 29.1ms = **1.28x** — but this omits the replicated layernorms/residual-adds/
  s-path, which also pay the ~1.5× tax.

### Full size sweep (warm per-call, ms; speedup = single/mesh)

| L       | trimul        | triattn       | transition    | per-layer z-path |
|---------|---------------|---------------|---------------|------------------|
| 256     | 2.84→1.92 **1.48x** | 1.40→1.74 **0.81x** | 1.90→4.97 **0.38x** | 10.4→12.3 **0.84x (LOSS)** |
| 512     | 9.69→5.62 **1.72x** | 6.46→5.19 **1.25x** | 4.88→7.52 **0.65x** | 37.2→29.1 **1.28x** |
| 686→704 | **OOM** (L1)  | (n/a)         | (single 9.10) | **OOM** |

(per-layer z-path = 2×trimul + 2×triattn + transition; omits replicated norms/adds/s-path)

## Verdict: head-sharding triangle-attention does NOT rescue trunk-TP — DEAD END

The untested lever turns out to scale exactly like trimul — it helps only at large L
(0.81x@256, 1.25x@512) — and three independent blockers remain, now quantified:

1. **Small-L net LOSS.** @256 the per-layer z-path is **0.84x** (mesh slower) *before*
   even counting the diffusion. Driven by the mesh-dispatch tax on the cheap replicated
   ops: transition collapses to **0.38x@256 / 0.65x@512**, and the same ~1.5× tax hits
   the 7 layernorms + 5 residual adds + s-path per layer. At small L the sharded ops are
   too cheap for the 1.4–1.7× gains to cover the replicated-op tax + CCL launches.
2. **Large-L OOM.** @704 the mesh trimul **TT_FATALs Out-of-Memory in L1** (126 MB L1
   buffer can't fit alongside the CCL/fabric L1 reservation) — the exact L1-vs-all_gather
   clash flagged 2026-06-24 (worked around there with DRAM, at a speed cost). The ceiling
   size — where trunk-TP gains would be largest — is precisely where it breaks without a
   DRAM rework.
3. **Diffusion cross-context (unchanged blocker).** A trunk-only mesh needs the 200-step
   diffusion to stay single-device while the trunk runs on the 4-mesh — two device
   contexts with weights resident on both. Running diffusion *replicated on the mesh*
   instead (the simpler option) doubles it via the per-step mesh _to_torch/from_torch
   tax (the 06-24 whole-model killer; consistent with transition's 0.65x here).

Op-level shards are real and ~bit-identical (trimul maxdiff=0; triattn head-shard
maxdiff 2.4e-3, PCC≈1), but they win only in the L=512 sweet spot, lose at 256, OOM at
686, and cannot reach an e2e win without solving the diffusion cross-context problem.
**Recommendation: ABANDON trunk tensor-parallelism for Boltz-2 `--fast`.** The op-level
TP win is genuine but cannot be assembled into a robust all-sizes e2e win on this 4-card
ring. No live-model code was changed; nothing to revert; accuracy untouched.

## Branch / artifacts
- Branch `exp/perf-20260630-trunk-profile` — only adds `.perf-runs/20260630/` (spike +
  this report). No `tt_bio/` edits.
- Fresh warm baseline (parallel sweep, host-contention-inflated ~+1s on diffusion) matched
  the documented main ceiling: 256f 16.2s / 512f 31.6s / 686f 51.7s / 512default 40.2s.

