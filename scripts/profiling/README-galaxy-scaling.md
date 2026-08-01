# Galaxy concurrency scaling — how to run the sweep, and what it measured

`galaxy_conc_sweep.py` answers "how close to N-times throughput do N concurrent single-card folds
get, and what stops them" on a many-chip single-host box. It launches N identical folds, one pinned
chip each, and samples host and per-fold CPU at 1 Hz, so a throughput number always arrives with the
CPU accounting that explains it. It imports no ttnn and never opens a device itself.

## Result on the 32-chip Wormhole galaxy (UF-EV-A13-GWH02)

One target (`abag_xm/9d3j`, `--diffusion_samples 1`, `--host_threads 2`), one chip per fold:

| N | cell wall | folds/hour | speedup | efficiency | host busy cores | cores/fold |
|---|---|---|---|---|---|---|
| 1  | 337 s   | 10.7  | 1.00x  | 100%  | — | — |
| 8  | 343 s   | 84.0  | 7.86x  | 98.2% | — | — |
| 16 | 352 s   | 163.6 | 15.32x | 95.7% | — | — |
| 24 | 367 s   | 235.4 | 22.04x | 91.8% | — | — |
| 28 | 378.8 s | 266.1 | 24.92x | 89.0% | 29.09 | 1.013 |
| 32 | 388.9 s | 296.2 | 27.72x | 86.7% | 33.28 | 1.017 |

Rows 1-24 are the campaign's own `conc.sh` run, rows 28/32 are this harness. 32 chips deliver
**27.7x**. The limiter is host physical cores: each concurrent fold costs ~1 host core, and this box
has 32 physical cores for 32 chips, so at N=32 the folds ask for 32.5 cores from a 32-core host.
About half of that core is compressible spin — four folds confined to 2 physical cores still run at
91% speed — so the irreducible host work is ~0.3-0.5 cores per fold.

## Running it

```
python3 galaxy_conc_sweep.py --yaml examples/abag_xm/9d3j.yaml --model opendde-abag \
    --msa-dir ~/abag_xm/msa_cache --out-root /tmp/g32 --jsonl /tmp/g32/cells.jsonl \
    --levels 32,28,24,16,8,4,2,1 --chip-order spread --host-threads 2
```

`--pin-sweep M1,M2,...` instead of `--levels` holds concurrency at `--level` and confines every fold
to `M` physical cores (both SMT siblings of each), which separates "the host cores are spinning on
dispatch back-pressure" from "the host is doing real work". `--chips a,b,c` takes an explicit chip
list, for when another job owns part of the box.

Four things that will otherwise cost you a pass:

* **The box must be yours.** If a serve pool or another campaign holds the chips, every fold blocks
  on device open and the cell returns `ok: 0` — filter on `ok > 0` when reading the JSONL.
* **Run it detached with `trap "" INT HUP`.** `setsid nohup` alone is not enough; a run died mid-cell
  on a SIGINT it should never have received.
* **Discard the first cell of a session** — cold JIT/program cache is ~10x.
* **Count folds, not processes.** One fold is a wrapper, a parent, a fork child that owns the device,
  and spawn grandchildren; counting processes halves per-fold CPU and inverts the conclusion.
