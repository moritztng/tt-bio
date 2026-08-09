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
