# ESMC multi-card wall-clock scaling (measured, on-hardware)

Follow-up to the ESMC embed data-parallel fanout (`--devices`, bit-exact sharding
parity already verified — see `scripts/esmc_multicard_parity.py`). That work only had
one leased card, so true *concurrent* multi-card speedup was never measured. This
measures it on qb1 (4x Blackhole p150a, all cards free).

## Method

Warm wall-clock, same fixed input set re-sharded across `--devices 0`, `0,1`, `0,1,2`,
`0,1,2,3` (`batch_size=8`, `--format parquet`). One measurement per point (not averaged
over repeats — see caveats). `esmc-6b` is the weight-bound target the sharding work
flagged as the best fanout candidate; `esmc-600m` is the contrast (smaller weights,
batched/bucketed compute path).

## Results

| model | N sequences | 1 card | 2 cards | 3 cards | 4 cards |
|---|---|---|---|---|---|
| esmc-600m | 256   | 14.5s (1.00x) | 14.7s (0.98x)  | 23.2s (0.62x) | 15.5s (0.93x) |
| esmc-600m | 4096  | 60.8s (1.00x) | 43.9s (1.38x)  | 35.1s (1.73x) | 31.0s (1.96x) |
| esmc-6b   | 48    | 43.5s (1.00x) | 40.9s (1.06x)  | 50.6s (0.86x) | 52.8s (0.82x) |
| esmc-6b   | 256   | 89.7s (1.00x) | 66.8s (1.34x)  | 123.0s (0.73x)| 136.0s (0.66x)|

(speedup vs 1-card, same total N; `d2`/`d3`/`d4` = number of cards)

## Verdict: mixed, and the 6B case is a genuine regression, not just "no win"

**esmc-600m/N=4096 is the one clean win**: sub-linear but real, ~2x at 4 cards. Enough
compute per shard (4096/4 = 1024 seqs) to amortize the per-worker fixed cost.

**Everything else is flat or actively worse with more cards**, including the 6B model
the sharding work predicted would fanout best. That prediction assumed weight-bound ⇒
near-linear, but it missed that *loading* the weights is itself the expensive, and
partly serialized, part for a 6B model:

- **esmc-600m/N=256**: workload too small at any device count — compile + fixed
  per-worker overhead dominates (`esmc-embed-batching` already found single-card warm
  throughput needs a bs≥8 steady state to pay off; 256/N-cards seqs never gets there
  once N-cards > 1).
- **esmc-6b/N=48 and N=256**: 2 cards gives a small real win (1.06x, 1.34x), but 3-4
  cards is *worse than 1 card*. Fitting the single-card numbers (43.5s@48, 89.7s@256)
  to a fixed-cost + per-seq-cost line gives per-seq ≈0.22s and a serial fixed cost
  (weight load + device init + program-cache warm) of ≈33s. If that fixed cost were
  unaffected by concurrency, 4-way sharding N=256 (64 seqs/shard) should land near
  33 + 64·0.22 ≈ 47s. It actually takes 136s — so the fixed cost itself is inflating
  with concurrency, not just failing to amortize. Back-computing the effective
  per-shard fixed cost at each device count (subtracting the shard's own compute):
  ~33s (d1, serial) → ~38s (d2) → ~104s (d3) → far worse (d4). The blowup is
  non-linear between 2 and 3 concurrent loads, consistent with contention loading/
  pushing multiple ~6B-parameter weight sets to the device fleet at once (PCIe
  bandwidth and/or the shared device-open lock noted in `device-open-lock-fleet-
  deadlock` — not root-caused further here, out of scope for this measurement pass).

## Takeaway

`--devices` fanout is a real win only when a shard's own compute is large relative to
its model's load/init cost — true here for 600M at N=4096, not for 6B at these sizes,
and not for either model at N=256 or below. **`--devices` on `esmc-6b` for small-to-
medium batches can make wall-clock *worse*, not better** — the README note has been
softened accordingly. No correctness impact: sharding is still bit-exact regardless of
speedup (see the parity harness); this is a performance-only finding.

## Caveats

Single-run measurements (no repeats) on a shared-but-idle host — the 600M/256 numbers
in particular bounce around (14.5/14.7/23.2/15.5s) enough that individual points have
real noise; the *shape* (600M/4096 scaling, 6B degrading past 2 cards) is the
consistent signal, not any single number. If this becomes decision-relevant (e.g.
choosing a default `--devices` policy), re-run with 3+ repeats per point.
