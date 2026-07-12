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

---

## Fix: persistent-worker embed (`--controller`)

Separately from the `--devices` fanout fixes above (Fix 1/Fix 2, which make concurrent
per-call subprocess fanout itself scale), the *per-call cold reload* is a distinct cost
that `--controller` eliminates entirely for repeated/production use — see below.

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

---

# esmc-300m/esmc-600m re-measured post thread-cap fix (2026-07-10, qb2)

Follow-up to Fix 2's own caveat: the 4-card CPU-thrash signature it diagnosed for
`esmc-6b` was also visible in the *original* table above for `esmc-600m/N=256`
(anomalous 3-card 0.62x point), but that row was never re-measured with the fix
applied. This closes that gap — same method as the original table (fixed
`make_seqs`-generated input set, `--batch_size 8`, one measurement per point, warm
compile cache), same host (qb2, 16 cores) as Fix 1/Fix 2, now covering both smaller
ESMC variants across all three N values Fix 2 flagged as unmeasured (48/256/4096).

**Method note — two deviations from the original table, neither expected to affect
timing:** (1) `--format npz` instead of `--format parquet` — this qb2 env doesn't have
`pyarrow`/`fastparquet` installed (not a declared dependency), and output writing is a
small fraction of wall-clock either way. (2) **Newly hit and worked around**: on qb2
`tt-bio embed --devices` with >1 device TT_FATALs at device-open
(`is_custom_fabric_mesh_graph_desc_path_specified`) unless `TT_MESH_GRAPH_DESC_PATH`
points at `p150_mesh_graph_descriptor.textproto` — this is the same qb2 P300-board-
misdetection quirk already documented for `predict`/TT-Atom, but `embed`'s fanout path
never had it wired in (predict's `_build_worker_device_assignments` sets it per
worker; `esmc._spawn_shard` doesn't). Worked around here by exporting the env var only
for the >1-device runs (the single-device path opens the chip directly, outside a
mesh, and TT_FATALs *if* the descriptor is set). **This means `tt-bio embed --devices`
is currently broken out-of-the-box on qb2 for >1 card** — a real gap, not something
this task's scope includes fixing; tracked as a follow-up below.

## Results

| model | N | 1 card | 2 cards | 3 cards | 4 cards |
|---|---|---|---|---|---|
| esmc-300m | 48   | 7.9s (1.00x)   | 7.9s (1.00x)   | 8.5s (0.94x)   | 9.0s (0.88x)   |
| esmc-300m | 256  | 15.7s (1.00x)  | 13.9s (1.13x)  | 19.3s (0.81x)  | 14.8s (1.06x)  |
| esmc-300m | 4096 | 133.0s (1.00x) | 121.6s (1.09x) | 119.7s (1.11x) | 119.6s (1.11x) |
| esmc-600m | 48   | 9.1s (1.00x)   | 9.4s (0.96x)   | 9.9s (0.92x)   | 16.1s (0.57x)  |
| esmc-600m | 256  | 18.6s (1.00x)  | 16.4s (1.14x)  | 21.4s (0.87x)  | 17.5s (1.06x)  |
| esmc-600m | 4096 | 163.3s (1.00x) | 147.7s (1.11x) | 144.8s (1.13x) | 143.6s (1.14x) |

(esmc-600m/48/1-card was measured twice: a cold-cache 97.3s first pass — the first
`esmc-600m` compile on this host during this sweep — and a 9.1s warm re-run once the
compile cache was warm, exactly the compile-dominated/disk-cached pattern
`esmc-embed-batching` already found. The warm number is what's tabulated, matching
the "warm wall-clock" method every other point in this doc uses.)

## 1. Is the 600m/N=256 3-card anomaly gone?

**Yes, the severe version is gone.** The original table's 3-card point was 23.2s vs
14.5s(1card)/14.7s(2card) — a 0.62x cliff, roughly 1.6x *slower* than its neighbors.
Re-measured here: 21.4s vs 18.6s(1card)/16.4s(2card) — 0.87x, a mild dip fully inside
the noise band this doc's own caveats already flagged ("600M/256 numbers... bounce
around (14.5/14.7/23.2/15.5s) enough that individual points have real noise"). No
point in either model at any N showed a repeat of a >1.5x cliff at 3 cards. Given
single-run methodology (no repeats, same limitation as the original table), it isn't
possible to fully separate "Fix 2 helped" from "the original 0.62x was itself just an
unlucky noisy sample" — but the *qualitative* regression signature the fix targeted
(severe, isolated 3-card cliff) did not reproduce.

## 2. Regression check for previously-documented configurations

**No regression.** `esmc-300m` was never in the original wall-clock table (only used
as the `esmc_multicard_parity.py` default model), so there's nothing prior to compare
it against — these are new baseline numbers, not a re-check. For `esmc-600m/256` and
`esmc-600m/4096`, the only rows that were previously measured (on **qb1**, not qb2 —
the original table's header says so explicitly), the qb2 numbers are real but far more
modest than qb1's: `600m/4096` scales only to 1.11-1.14x here vs qb1's ~2x, and
`600m/256` shows small (~1.1x) wins/dips instead of qb1's noisy-but-roughly-flat
shape. This is **not** a regression caused by the thread-cap fix — Fix 2 only changes
host env vars in the shard subprocess, it cannot make things slower, and no config
here is worse than its own 1-card baseline by more than the noise band already
documented. The gap vs qb1's numbers is a cross-host difference, most plausibly the
newly-found mesh-descriptor gap above: every >1-device shard subprocess on qb2 pays a
P300-workaround control-plane/mesh init cost that qb1 (real P150a boards, no
misdetection) never pays. That cost is negligible next to `esmc-6b`'s 10-16s weight
load (hence 6b fanout still scales cleanly on qb2, per Fix 2 above) but is
comparable to `esmc-300m`/`esmc-600m`'s much smaller per-shard load/init time — capping
their multi-card win on this host specifically. Not root-caused further here (would
need per-shard phase timing, `TT_BIO_TIMING`, to confirm); flagged as a follow-up.

One new small-N anomaly, same known cause as the rest of this doc: `esmc-600m/N=48`
at 4 cards is 0.57x (16.1s vs 9.1s@1card) — 12 seq/shard is too little compute to
amortize the per-shard fixed cost (now inflated further by the mesh-descriptor
overhead above), the same "fanout doesn't pay off below some N" pattern already
documented for small batches elsewhere in this doc. Not a fanout-fix regression.

## 3. Parity — SACRED, re-verified

`scripts/esmc_multicard_parity.py --n 24 --shards 4` (with `TT_MESH_GRAPH_DESC_PATH`
set, see method note): **PASS, bit-exact** (`Δmax per_residue=0 pooled=0`) for both
`esmc-300m` and `esmc-600m`. Thread-pool sizing and the mesh-descriptor env var are
both host/env-only changes — device numerics are unaffected, as expected.

---

# Fix 3 (2026-07-12, qb2): `embed` opens its own P300 mesh-graph descriptor

The follow-up flagged above is now fixed: `esmc._spawn_shard` detects a P300-
misdetected board and sets `TT_MESH_GRAPH_DESC_PATH` for its subprocess itself
(same `main._detect_p300_devices` / `_find_ttnn_mesh_graph_descriptor` predict
and `boltzgen gen` already use — no second detection method). `tt-bio embed
--devices` with 2/3/4 cards now runs clean on qb2 with no manual env var.

**A wider gap than originally scoped**: the single-card, no-`--devices` path
(`embed_cmd`'s in-process branch, `tt_bio/main.py`) turned out to be broken the
same way — it opens the device directly without ever going through predict's
worker-assignment code, so it never set the descriptor either. Confirmed by
direct reproduction: `TT_VISIBLE_DEVICES=<any card>` alone TT_FATALs on this
host with the exact same "Custom fabric mesh graph descriptor path must be
specified" error, on device 0, 1, and 3 — this is a host-wide firmware quirk
(all 4 qb2 cards misreport as P300), not specific to any one card or to
multi-card fanout. Fixed the same way in `embed_cmd`. So `tt-bio embed`, single
or multi-card, now works out of the box on qb2 with no env var at all.

(A previous doc note above claimed the single-card path "TT_FATALs *if* the
descriptor is set" — that does not reproduce now; setting it unconditionally
for a single-chip open on this host succeeds cleanly. Superseded.)

## Parity, re-verified after the real fix (no manual workaround)

`scripts/esmc_multicard_parity.py`, `TT_MESH_GRAPH_DESC_PATH` unset (relying
purely on the code fix): **PASS, bit-exact** for both `esmc-300m` (`--n 24
--shards 4`) and `esmc-600m` (`--n 16 --shards 4`), `Δmax per_residue=0
pooled=0`. Same result as the workaround-assisted run above, as expected — the
fix only changes *when* the env var gets set, not any compute path.

## Does fixing the gap close qb2's scaling shortfall vs qb1? No — re-measured, real gap remains

Re-ran the wall-clock curve with the fix in place and zero manual workaround
(warm cache, `batch_size=8`, `--format npz`, one run per point — same
single-run-per-point caveat as the rest of this doc):

| model | N | 1 card | 2 cards | 3 cards | 4 cards |
|---|---|---|---|---|---|
| esmc-300m | 256  | 13.9s (1.00x) | 14.8s (0.94x) | 15.2s (0.91x) | 16.1s (0.86x) |
| esmc-600m | 256  | 17.3s (1.00x) | 17.7s (0.98x) | 18.2s (0.95x) | 19.0s (0.91x) |
| esmc-600m | 4096 | 190.6s (1.00x)| 181.2s (1.05x)| 178.7s (1.07x)| 177.3s (1.07x)|

`esmc-600m/4096` is the config this doc's earlier remeasurement flagged as
possibly capped by the mesh-descriptor overhead (qb2 1.11-1.14x vs qb1's
~2x). With the fix applied — and now paying the descriptor-init cost
*consistently* at every card count, including 1 card, instead of only at
&gt;1 card as the manual-workaround measurement above did — the scaling
barely moves (1.05-1.07x). **The hypothesis was wrong: removing the gap did
not recover qb1-like scaling.** The qb2-vs-qb1 shortfall is a real,
still-unexplained host difference (plausibly core count / PCIe topology /
per-shard host overhead — not root-caused further here), not an artifact of
the missing mesh descriptor. `esmc-300m`/`esmc-600m` at N=256 show no
material multi-card win on qb2 either way, consistent with this doc's
original finding that fixed per-shard overhead dominates below some N.

**Practical takeaway**: the correctness gap (crash without a manual env var)
is fully closed. The performance gap (qb2 fanout scaling well below qb1's) is
not related to it and remains open — `--devices` fanout on qb2 is a much
smaller win than on qb1 for these two smaller ESMC variants; `esmc-6b`
(Fix 1/Fix 2 above) is unaffected by any of this and still scales to 1.49x
@ 4 cards.

