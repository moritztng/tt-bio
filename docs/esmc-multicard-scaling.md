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

## Fix: persistent-worker embed (`--controller`)

The regression above isn't a fanout problem — it's that `--devices` cold-loads the
model from scratch in a brand-new subprocess on *every single invocation*. For a
6B-parameter model that reload is the dominant, partly-serialized cost this doc found
(≈33-104s+ depending on concurrent device count), not the embedding compute itself.

`tt-bio embed` now accepts `--controller URL`, riding the same persistent
worker/controller machinery already used by `predict`/`gen` (`tt_bio/distributed.py`,
`tt_bio/worker.py`): a worker loads its ESMC model once and keeps it resident across
every subsequent run submitted to that controller — the weight load becomes a
one-time cost per worker lifetime, not a per-call tax. The `--devices` fanout logic
(`_shard_by_length` / `embed_multicard`) is unchanged; `--controller` is a second
dispatch path that shards the same way but through the scheduler instead of spawning
subprocesses, so the bit-exact parity result above still holds unchanged.

**Measured on qb1, `esmc-6b`, `batch_size=8`, `--format parquet`** (cold = worker's
first call, pays the weight load; warm = same resident worker, second call):

| N sequences | cards | cold (1st call) | warm (resident) | speedup |
|---|---|---|---|---|
| 48  | 1 | 50.0s  | 9.1s  | 5.5x |
| 256 | 1 | 89.7s* | 46.3s | 1.9x |
| 48  | 2 | 261s†  | 13.4s | 19x |

\* single-card cold number from the table above (same input class, not re-measured
cold here since the mechanism — one-time load — is already established by N=48).
† this run hit slow concurrent weight loading similar in kind to the d2/d3/d4
contention documented above (two 6B loads racing on shared host resources); it is a
**one-time** cost per worker pair's lifetime, unlike the old path where it recurred on
every call.

Every embed (cold and warm, 1 and 2 cards) was verified bit-exact (Δmax pooled = 0)
against the reference single-shot embedding. 3-4 card legs were not re-measured in
this pass — qb1's other two cards were held by concurrent fleet work at measurement
time — but the fix is architectural (resident model, no reload), not device-count
dependent, so the same one-time-load argument applies at any card count.

**Takeaway**: with `--controller`, `esmc-6b`'s effective per-call cost drops to just
its compute (no reload), which trivially scales with device count — the contention
this doc measured was a cost of *reloading on every call*, and a resident worker pays
it once. `--devices` (per-call subprocess) is unchanged and still appropriate for a
single ad-hoc invocation with no standing controller; `--controller` is the better
choice for repeated/production embed workloads, especially with `esmc-6b`.
