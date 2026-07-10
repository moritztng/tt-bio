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

---

# Fix (2026-07-10, qb2): shared `/dev/shm` tiled-weight cache

The 6B regression above was root-caused to **redundant per-worker weight load**. In
`embed_multicard` each of the N pinned card-workers independently (a) read the 24 GB
checkpoint, (b) converted it to the device dtype, and (c) tiled it on host via
`from_torch`. Steps (a)+(b)+(c) are host-memory-bandwidth bound and *identical* across
workers, so running N of them at once just contends — per-worker load grows ~linearly
with N, and because 6B compute is tiny (~0.22 s/seq once kernels are disk-cached),
fanout is essentially all load ⇒ more cards = more contention = slower.

## Measured contention (qb2, warm page cache)

Host-only concurrent load (disk read + fp32→bf16 convert, **no device**), N processes:

| N | per-worker load |
|---|---|
| 1 | 1.5 s |
| 2 | ~3.1 s |
| 4 | ~6.7 s |

Single-card upload (card 0): `from_torch(device)` over all 802 weights = **4.49 s**;
the host tiling inside it is the same redundant, contention-prone work.

## The fix

`load_esmc6b_shared` + `tenstorrent.weight_cache`: the first worker to arrive (the
builder, chosen by an `flock` on the cache dir) tiles every weight on host **once** and
publishes each tile to a `/dev/shm` dir (`ttnn.dump_tensor`, atomic temp+rename); peers
block on the lock until `.done`, then `ttnn.load_tensor` each pre-tiled weight straight
to their own card (no checkpoint read, no re-tiling) and pay only the per-card DMA,
which runs in parallel across the independent PCIe links. Keyed by the torch_to_tt call
index — stable because every worker constructs the identical module in the identical
order. Enabled only for `esmc-6b` fanout (`--devices` with >1 card); all other models
and the single-card path are byte-for-byte unchanged (`_weight_cache is None`).

## Per-worker load cost (card 0, esmc-6b, warm)

| path | cost |
|---|---|
| builder (`dump`: tile + dump + to_device, one-time) | 10.53 s |
| peer (`load`: load_tensor + to_device) | **2.22 s** |
| *old* (each worker: disk+convert + `from_torch`) | ~10–16 s, ∝N contended |

So fanout wall-clock becomes ≈ one 10.5 s build + a ~2.2 s parallel DMA per card
(flat in N) instead of N contending ~10–16 s loads — turning the regression into
flat/improving scaling.

## Parity — SACRED, verified bit-exact

8 sequences × 192 aa, esmc-6b, non-fast, card 0. `max|Δ| = 0.000e+00` (per-residue and
pooled) for **build vs single-card baseline**, **load vs baseline**, and **load vs
build**. A dumped tile is exactly what `from_torch` would have produced, so the device
tensors are identical.

## Honest status — end-to-end 4-card curve pending free cards

The component wins above are measured on card 0, and the design makes load one-time +
parallel-DMA by construction. But the **end-to-end 1/2/4-card wall-clock re-measurement
on qb2 was not run**: 3 of qb2s 4 cards were held by an active fleet worker

---

# Fix (2026-07-10, qb2): shared `/dev/shm` tiled-weight cache

The 6B regression above was root-caused to **redundant per-worker weight load**. In
`embed_multicard` each of the N pinned card-workers independently (a) read the 24 GB
checkpoint, (b) converted it to the device dtype, and (c) tiled it on host via
`from_torch`. Steps (a)+(b)+(c) are host-memory-bandwidth bound and *identical* across
workers, so running N at once just contends — per-worker load grows ~linearly with N,
and because 6B compute is tiny (~0.22 s/seq once kernels are disk-cached) fanout is
essentially all load ⇒ more cards = more contention = slower.

## Measured contention (qb2, warm page cache)

Host-only concurrent load (disk read + fp32→bf16 convert, **no device**), N processes:

| N | per-worker load |
|---|---|
| 1 | 1.5 s |
| 2 | ~3.1 s |
| 4 | ~6.7 s |

Single-card upload (card 0): `from_torch(device)` over all 802 weights = **4.49 s**;
the host tiling inside it is the same redundant, contention-prone work.

## The fix

`load_esmc6b_shared` + `tenstorrent.weight_cache`: the first worker to arrive (the
builder, chosen by an `flock` on the cache dir) tiles every weight on host **once** and
publishes each tile to a `/dev/shm` dir (`ttnn.dump_tensor`, atomic temp+rename); peers
block on the lock until `.done`, then `ttnn.load_tensor` each pre-tiled weight straight
to their own card (no checkpoint read, no re-tiling) and pay only the per-card DMA,
which runs in parallel across the independent PCIe links. Keyed by the `torch_to_tt`
call index — stable because every worker constructs the identical module in the
identical order. Enabled only for `esmc-6b` fanout (`--devices` with >1 card); all other
models and the single-card path are byte-for-byte unchanged (`_weight_cache is None`).

## Per-worker load cost (card 0, esmc-6b, warm)

| path | cost |
|---|---|
| builder (`dump`: tile + dump + to_device, one-time) | 10.53 s |
| peer (`load`: load_tensor + to_device) | **2.22 s** |
| *old* (each worker: disk+convert + `from_torch`) | ~10–16 s, ∝N contended |

So fanout wall-clock becomes ~one 10.5 s build + a ~2.2 s parallel DMA per card (flat
in N) instead of N contending ~10–16 s loads — turning the regression into
flat/improving scaling.

## Parity — SACRED, verified bit-exact

8 sequences × 192 aa, esmc-6b, non-fast, card 0. `max|Δ| = 0.000e+00` (per-residue and
pooled) for **build vs single-card baseline**, **load vs baseline**, and **load vs
build**. A dumped tile is exactly what `from_torch` would have produced, so the device
tensors are identical. Harness: `scripts/esmc6b_shared_cache_parity.py`.

## Honest status — end-to-end 4-card curve pending free cards

The component wins above are measured on card 0, and the design makes load one-time +
parallel-DMA by construction. But the **end-to-end 1/2/4-card wall-clock re-measurement
on qb2 was not run**: 3 of qb2's 4 cards were held by an active fleet worker
(tt-bio worker --device_ids 1,2,3) for the whole session. Re-run the table at the top
of this doc, with the fix, on 4 free cards before treating the scaling as final.
