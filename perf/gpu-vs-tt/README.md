# Protenix-v2 / OpenDDE — Tenstorrent vs NVIDIA

Scripts for a like-for-like latency comparison of the same models and the same weights on a
Blackhole p150 (tt-bio) and on an NVIDIA GPU (upstream Protenix / OpenDDE with cuEquivariance).

The full benchmark contract — parameters, warm-up rules, the correctness gate, what was
deliberately left out — lives in `~/.coworker/state/gpu-vs-tt-protenix-opendde-benchmark.md`.
Read it before running anything; the numbers only mean something under those conditions.

## Files

| File | Runs on | What it does |
|---|---|---|
| `make_inputs.py` | any host | Turns one tt-bio YAML into matched inputs for both stacks, sharing a single MSA file |
| `gpu_setup.sh` | rented GPU box | Unattended, version-pinned install of Protenix + OpenDDE + cuEquivariance + weights |

## Throughput at concurrency

`../../scripts/gpu_vs_tt/` measures both sides as throughput, not only latency: N concurrent
folds on one device, one process per fold, with a shared barrier and the same aggregate
estimator on both sides (`conc.py`). `tt_concurrency.py` is the Tenstorrent leg (one process
per card; a second process on the same chip is refused by the device lease, so per-card
concurrency is 1 by construction). `gpu_concurrency.py` driven by `gpu_conc_session.sh` is the
NVIDIA leg (plain concurrent processes, then the same sweep under MPS).

The fairness rule: each side is quoted at its own best intra-device concurrency, measured with
the same estimator, and one device vs one device is the headline unit. Per-box numbers are
secondary and labelled as such.

Two flags exist to check that a throughput number means what it looks like:

`gpu_concurrency.py --targets a,b,c,...` gives worker i its own target out of
`scripts/gpu_vs_tt/fixtures/distinct/` instead of every worker folding a copy of the same one.
Use it whenever a concurrency number is going to be quoted: an N-way point where all N workers
fold the same input cannot, on its own, rule out that the throughput came from the repetition.

`tt_concurrency.py --hoist` times `model.fold` only, featurizing once up front and suppressing
the CIF write, which is the boundary the GPU leg times. `--instrument` keeps the normal
`predict_one` boundary and records the per-fold featurize/fold/write split instead. Between
them they answer how much of a TT fold is host work rather than device work. Either way every
timed fold's pLDDT is checked against the cold fold's, so a fold that ran on mutated or emptied
features fails loudly instead of returning a fast wrong number.

## Quick start

Populate the MSA cache once (free, on any host with network), then build the inputs:

```bash
python3 make_inputs.py --yaml ../../examples/615.yaml \
    --msa-cache ~/.boltz/bench_msa --out-dir ./inputs/T615 --repeats 6 --verify
```

`--repeats 6` means one discarded warm-up fold plus five timed ones, in a single process. Both
sides use the same count.

On the GPU box:

```bash
bash gpu_setup.sh 2>&1 | tee setup.log
```

Then run the ladder from the state doc. Tear the instance down when finished and confirm with
`vastai show instances` that it is actually gone — the destroy command can silently fail.
