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

Host CPU per fold is constant across the whole range (381 core-seconds at N=1, 394 at N=32, +3.4%)
while the wall grows 15.2%, so the extra time at high N is time folds spend waiting rather than
computing on the host. What is left to test is host DRAM/IO bandwidth, or a device-side effect this
box cannot profile (the pip ttnn wheel has no Tracy build).

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

* **The box must be yours.** If a serve pool or another campaign holds the chips, every fold blocks
  on device open and the cell returns `ok: 0` — filter on `ok > 0` when reading the JSONL. Load
  average is the quick tell: ~1 per fold means computing, near zero means blocked.
* **Run it detached with `trap "" INT HUP`.** `setsid nohup` alone is not enough; a run died mid-cell
  on a SIGINT it should never have received.
* **Discard the first cell of a session** — cold JIT/program cache is ~10x.
* **Count folds, not processes.** One fold is a wrapper, a parent, a fork child that owns the device,
  and spawn grandchildren; counting processes halves per-fold CPU and inverts the conclusion.
