# ESMFold2 bf8 trunk-weight scout — no-go (speed-neutral, accuracy-safe)

**Verdict:** routing ESMFold2's 48-block trunk `TriangleMultiplication` matmul
weights through `tenstorrent._dtype()` (block-fp8 under `--fast`) is **speed-neutral
and reverted**. The accuracy path is fine. The lever the scout bet on (weight-read
bandwidth) is not the trunk's bottleneck.

## The gap that was real

ESMFold2's trunk tri_mul weights stayed bf16 even under `--fast`
(`tt_bio/esmfold2.py`, `_DTYPE = bfloat16`), while Boltz-2 and Protenix-v2 already
drop their trunk weights to bf8 via the shared `tenstorrent._dtype()` helper. The
mechanism existed; ESMFold2's trunk just never routed through it. The trunk is the
dominant warm stage, so this looked like the one remaining single-card precision lever.

## What was tried

`PairUpdateBlock` passed `weight_dtype=_dtype()` to `TriangleMultiplication` so the
two tri_mul matmul weights (`g_out`, `p_out`, and the per-pair projection) drop to
block-fp8 under `--fast`, mirroring the pair-transition `SwiGLUFFN` and the
Boltz-2/Protenix attention/FFN precedent. Weight-storage change only; activations and
matmul accumulation were left untouched.

## Measured A/B

Blackhole card 0, qb2, `scripts/kernel_scout_next_bench.py`, 48-block trunk, both legs
under `--fast` (bf8-weights vs bf16-weights), `--skip-sync-profile`.

| N    | bf8 weights (s) | bf16 weights (s) | speedup |
|------|-----------------|------------------|---------|
| 512  | 3.8564          | 3.8576           | 1.000x  |
| 1024 | 19.2011         | 19.2576          | 1.003x  |

Both within run-to-run noise. Per-component host-enqueue at N=1024: trimul_in 6.27s,
trimul_out 6.93s, transition 4.01s. The tri_mul pair tensor is `[N, N, 128]` and
dominates DRAM bandwidth; the tri_mul weights are small and SRAM-resident, so halving
their storage does not relieve the bottleneck. The one genuinely weight-bound trunk op
(the pair-transition) already loaded `fc1`/`fc2` at `_dtype()`, and its matmul compute
was already bf8 under `--fast`. This is the same dead-end shape as Boltz-2's trunk
(`docs/boltz2-protenix-kernel-scout.md`): activations already bf8, weights tiny vs the
pair tensor, keep fp32 accumulation.

## Accuracy gate (the path is safe even though it is not useful)

`scripts/release_gate.py --model esmfold2 --fast` (7ROA / `examples/prot.yaml`, 200
steps, 5 samples, best-by-confidence, ground-truth Kabsch RMSD + TM):

| model    | RMSD (A) | TM    | floor       | result |
|----------|----------|-------|-------------|--------|
| esmfold2 | 1.717    | 0.919 | <=4.0/>=0.65 | PASS   |

The bf8 weight path folds 7ROA to 1.72 A, comfortably inside the 4.0 A / 0.65 TM floor
and in line with the full-precision baseline (~2.2 A, within ESMFold2's seed noise).
A weight-storage dtype change with unchanged compute precision is numerically
near-neutral, as expected. The gate now has an opt-in `--fast` flag so the block-fp8
trunk path can be gated at all (it previously folded full-precision only).

## Outcome

- The weight-routing change is **reverted**. A 0.003x change does not earn the
  `weight_dtype` parameter and `_dtype` import it adds to the shared
  `TriangleMultiplication` class.
- Kept: the `--fast` / `--skip-sync-profile` flags on `scripts/kernel_scout_next_bench.py`
  (needed to measure large-N trunk wall-clock and the fast path at all) and the
  `--fast` flag on `scripts/release_gate.py` (the gate could not exercise the shipped
  fast path before).
- Single-card warm trunk precision is now exhausted across all four trunk models
  (Boltz-2, Protenix-v2, ESMFold2, ESMFold2-fast). Remaining ROI is cold-start and
  multi-device, not deeper weight quantization.
