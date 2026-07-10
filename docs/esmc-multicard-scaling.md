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

## Takeaway (superseded — see the two fixes below)

`--devices` fanout is a real win only when a shard's own compute is large relative to
its model's load/init cost — true here for 600M at N=4096, not for 6B at these sizes,
and not for either model at N=256 or below. **`--devices` on `esmc-6b` for small-to-
medium batches can make wall-clock *worse*, not better** — the README note has been
softened accordingly. No correctness impact: sharding is still bit-exact regardless of
speedup (see the parity harness); this is a performance-only finding.

*(This "no win at 3-4 cards" verdict was itself later found to have two separate,
independently-fixable causes — see below. It no longer holds after both fixes.)*

## Caveats

Single-run measurements (no repeats) on a shared-but-idle host — the 600M/256 numbers
in particular bounce around (14.5/14.7/23.2/15.5s) enough that individual points have
real noise; the *shape* (600M/4096 scaling, 6B degrading past 2 cards) is the
consistent signal, not any single number. If this becomes decision-relevant (e.g.
choosing a default `--devices` policy), re-run with 3+ repeats per point.

---

# Fix 1 (2026-07-10, qb2): shared `/dev/shm` tiled-weight cache

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

So the weight-*load* phase becomes ≈ one 10.5 s build + a ~2.2 s parallel DMA per card
(flat in N) instead of N contending ~10–16 s loads.

## Parity — SACRED, verified bit-exact

8 sequences × 192 aa, esmc-6b, non-fast, card 0. `max|Δ| = 0.000e+00` (per-residue and
pooled) for **build vs single-card baseline**, **load vs baseline**, and **load vs
build**. A dumped tile is exactly what `from_torch` would have produced, so the device
tensors are identical. Harness: `scripts/esmc6b_shared_cache_parity.py`.

## Fix 1 alone was not enough

Re-measuring the full end-to-end 1/2/3/4-card curve on qb2 with *only* the weight-cache
fix (N=256, same method as the original table): **1 card 138.4s, 2 cards 125.1s
(1.11x), 3 cards 167.8s (0.82x), 4 cards 156.8s (0.88x)**. Better than the pre-fix
curve (which regressed to 0.66x at 4 cards) but still *not* a monotonic win — 3-4
cards remained slower than 1 card. Instrumenting the load phase itself
(`TT_BIO_TIMING=/path`) confirmed the weight-cache fix was working exactly as designed
(builder 10.5s once, peers 12-16s `load_total` *flat* regardless of N, not the old
∝N blowup) — so the residual regression had to be coming from somewhere else in the
pipeline, not weight loading.

---

# Fix 2 (2026-07-10, qb2): cap per-shard host thread pools

## Root cause: host CPU oversubscription, not weight loading

Sampling `ps -eLo pid,pcpu,comm` every 10s during a 4-card run (post Fix 1) showed each
shard subprocess bursting to **200-380% CPU** during its compute phase, all four
concurrently — host `loadavg` peaked at **~21 on a 16-core host** (qb2), i.e. the 4
co-resident shards were oversubscribing the host by >5x. This is host-side
numpy/torch/BLAS thread-pool contention (each subprocess's OMP/MKL/OpenBLAS pool
defaults to *all* cores), not a device, PCIe, or weight-load bottleneck.

This turned out to be **the same bug already diagnosed and fixed once**, just not in
this code path: `tt_bio/main.py::_cap_worker_threads` caps exactly this for the fleet
worker-process pool (`_spawn_worker_processes`), with a comment describing this exact
symptom ("N co-resident workers spawn N*cores threads that thrash the CPU and collapse
throughput ... the multi-card slowdown"). `esmc.py`'s `embed_multicard` /
`_spawn_shard` — the `tt-bio embed --devices` fanout path — never had the equivalent
applied.

## The fix

`esmc._thread_cap_env(n_workers)`: mirrors `_cap_worker_threads` — caps
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS`
to `cores // n_workers` (operator-set values win), computed once in `embed_multicard`
and passed into every `_spawn_shard` subprocess's env. No change to device kernels or
numerics — purely a host thread-pool sizing knob.

## Result: N=256, same qb2 host, both fixes applied

| cards | wall-clock | speedup |
|---|---|---|
| 1 | 138.4s | 1.00x |
| 2 | 103.7s | 1.33x |
| 3 | 96.5s  | 1.43x |
| 4 | 93.1s  | **1.49x** |

Monotonically improving, as it should be for an embarrassingly-parallel, weight-bound
workload once both the weight-load and host-CPU contention sources are removed. (N=48,
the smaller batch from the original table, is expected to still show a weaker win —
fixed per-card DMA/init cost dominates at very small shards — but is no longer a
*regression*: not re-measured here, out of scope for this pass.)

## Parity — SACRED, re-verified after both fixes

- `scripts/esmc6b_shared_cache_parity.py --n 8 --len 192`: build vs baseline, load vs
  baseline, load vs build all `max|Δ|=0.000e+00`, bit-exact — PASS.
- `scripts/esmc_multicard_parity.py --model esmc-300m --n 24 --shards 4`: host-logic
  (shard coverage/balance/reassembly) OK, on-device Δmax=0 per-residue and pooled —
  PASS. (Thread-pool sizing is a pure env-var change; re-run to confirm it didn't
  perturb anything.)

## Takeaway

The `esmc-6b` `--devices` fanout regression had two independent causes, each masking
the other's fix from looking sufficient on its own: redundant weight loading (Fix 1)
and host CPU thread-pool oversubscription (Fix 2). With both applied, `esmc-6b` fanout
scales monotonically to 4 cards (1.49x @ N=256) — no longer a case where `--devices`
makes things worse. The README's earlier softened warning about `esmc-6b --devices`
should be revisited/removed now that this is fixed.

## Caveats

Single-run measurements per point (no repeats), one host (qb2), one N (256). The
N=48/N=4096/esmc-600m rows in the original table were not re-measured with Fix 2 and
may also improve (esmc-600m/N=256 in particular showed a similar host-CPU-thrash
signature, e.g. the anomalous 3-card 0.62x point) — worth a follow-up sweep if this
becomes decision-relevant.
