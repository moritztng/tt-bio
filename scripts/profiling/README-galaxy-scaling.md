# Galaxy concurrency scaling — how to run the sweep, and what it measured

`galaxy_conc_sweep.py` answers "how close to N-times throughput do N concurrent single-card folds
get, and what stops them" on a many-chip single-host box. It launches N identical folds, one pinned
chip each, and samples host and per-fold CPU at 1 Hz, so a throughput number always arrives with the
CPU accounting that explains it. It imports no ttnn and never opens a device itself.

## Result on the 32-chip Wormhole galaxy (UF-EV-A13-GWH02)

One target (`abag_xm/9d3j`, `--diffusion_samples 1`, `--host_threads 2`), one chip per fold, chips
interleaved across the four PCIe root complexes, all cells in one window with a same-session N=1
baseline:

| N | cell wall | folds/hour | folds/hour/chip | speedup | efficiency | cores/fold |
|---|---|---|---|---|---|---|
| 1  | 339.7 s | 10.60  | 10.598 | 1.00x  | 100%  | 1.122 |
| 2  | 340.2 s | 21.16  | 10.581 | 2.00x  | 99.8% | 1.100 |
| 4  | 341.2 s | 42.21  | 10.552 | 3.98x  | 99.6% | 1.085 |
| 8  | 343.2 s | 83.91  | 10.488 | 7.92x  | 99.0% | 1.073 |
| 16 | 353.4 s | 163.01 | 10.188 | 15.38x | 96.1% | 1.044 |
| 24 | 371.5 s | 232.58 | 9.691  | 21.94x | 91.4% | 1.019 |
| 28 | 378.8 s | 266.12 | 9.504  | 25.11x | 89.7% | 1.013 |
| 32 | 391.5 s | 294.28 | 9.196  | **27.76x** | **86.8%** | 1.006 |

**32 chips deliver 27.8x.** The curve is smooth — no knee, no collapse.

The remaining 13% is not host CPU, not chip placement, and not thread count:

* Confining all 32 folds to **16** physical cores costs 3.8% (to 8 cores, 20.4%), so about half of
  the ~1 core each fold consumes is compressible dispatch spin and the host has ~2x headroom at full
  concurrency.
* Putting 8 folds on a single PCIe root complex costs 0.15% versus spreading them over four — and
  that cell puts the same per-complex load on one complex that N=32 puts on all four, at a twelfth of
  the degradation.
* `--host_threads` 1 / 2 / 4 at N=32 spans 1.6%, trending *against* more threads.
* Host DRAM bandwidth is not close: one fold pulls 4.72 MB/s from DRAM (`perf stat -e
  ls_any_fills_from_sys.all_dram_io`, 20 s on a live fold), so 32 of them ask ~151 MB/s of the
  176.6 GB/s measured on this host — **0.09%**.

Host CPU per fold is constant across the whole range (381 core-seconds at N=1, 394 at N=32, +3.4%)
while the wall grows 15.2%, so the extra time at high N is time folds spend waiting rather than
computing — and not on any host resource above. What remains is on the device side or in the
host-device interconnect, which this box cannot see: the pip ttnn wheel has no Tracy build, so there
is no device profiler.

## The other fanout path: `tt-bio embed` through the serve pool

`embed_pool_probe.py` dispatches an embed through a live controller and samples at 4 Hz how many of
the resident pool workers are actually burning CPU, so "are the cards slow or idle" is answered by a
trace instead of inferred from the throughput gap.

They are idle. At N=2048 only 8.1 of 26 workers are busy on average (31%), and the busy trace shows
compute and result handling strictly serialized — a compute burst in the middle with the pool idle
before and after it.

The cause is the result payload, not the cards. Each sequence returns ~651 KB of per-residue
embeddings. Same N, same warm pool, A/B/A/B:

| format | wall | seq/s | output |
|---|---|---|---|
| npz (per-residue) | 27.92 / 27.83 s | 36.68 / 36.80 | 635 MB |
| parquet (pooled only) | 7.30 / 6.83 s | 140.32 / 149.97 | 7.7 MB |

**82x less result data, 4.1x faster, identical compute.** The parquet cells also put a floor under
the real compute ceiling: 150 seq/s across 26 workers is 5.8 seq/s per card, so any "per-card
seq/s" figure derived from the npz path was already transfer-limited rather than a compute number.

`embed_controller_probe.py` names the component. Sampling the controller process's own CPU next to
the pool's, the two curves are exact complements: on the npz path the pool is fully busy for the
first ~40% of the run and **completely idle for the last 50%**, while the controller pegs ~1.1-1.4
cores over exactly that tail. It spends **15 core-seconds of a 27 s wall** on result handling; the
parquet run does the same compute and needs **1.3**.

Every shard's results come back through one controller process as base64 inside JSON — 651 KB of
per-residue embeddings per sequence, inflated 33% by base64 — and the cards idle while that happens.
Removing the tail would take N=1024 from 27 s to ~12 s (~2.2x). Workers and client share a
filesystem here, so handing back a path instead of an 868 KB string is the obvious lever.

## Running it

```
python3 galaxy_conc_sweep.py --yaml examples/abag_xm/9d3j.yaml --model opendde-abag \
    --msa-dir ~/abag_xm/msa_cache --out-root /tmp/g32 --jsonl /tmp/g32/cells.jsonl \
    --levels 32,28,24,16,8,4,2,1 --chip-order spread --host-threads 2
```

`--pin-sweep M1,M2,...` instead of `--levels` holds concurrency at `--level` and confines every fold
to `M` physical cores (both SMT siblings of each), which separates "the host cores are spinning on
dispatch back-pressure" from "the host is doing real work". `--chips a,b,c` takes an explicit chip
list, for when another job owns part of the box or when you want every fold on one root complex.

Four things that will otherwise cost you a pass:

* **The box must be yours, and a stopped service does not mean it is.** If a serve pool or another
  job holds the chips, every fold blocks on device open and the cell returns `ok: 0` — filter on
  `ok > 0` when reading the JSONL. Load average is the quick tell: ~1 core per fold means computing,
  near zero means blocked. A stopped platform service usually means *someone else* has a maintenance
  window open and is about to restore it; taking the chips then blocks their restore and keeps the
  public site down. Deploy your own window, and arm a watchdog that restores it at a hard deadline
  whatever happens to your session.
* **Run it detached with `trap "" INT HUP`.** `setsid nohup` alone does not survive a dropped tunnel.
* **Discard the first cell of a session** — cold JIT/program cache is ~10x.
* **Count folds, not processes.** One fold is a wrapper, a parent, a fork child that owns the device,
  and spawn grandchildren; counting processes halves per-fold CPU and inverts the conclusion.
