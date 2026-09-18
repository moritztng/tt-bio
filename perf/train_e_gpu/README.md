# ABodyBuilder3's training step, measured on a rented GPU

What one optimizer step of upstream ABodyBuilder3 costs on one GPU, so the "comparable" in the
TRAIN done-definition has a measured referent instead of one derived from a paper's ">3x faster".

Nothing here runs on Tenstorrent hardware and nothing here imports `tt_bio`. It rents a GPU,
runs `Exscientia/ABodyBuilder3` unmodified, and reports seconds per step.

## The number

One step is one optimizer step at ABodyBuilder3's own global batch of 64 (8 micro-batches of 8
plus the update), which is the unit their released checkpoint's `global_step: 193,512` counts.

| | their `params.yaml` as shipped | dataloader overlapped |
|---|---|---|
| A100-80G | **6.994 s** | **3.016 s** |
| second A100-80G, independent rental | 7.119 s | 3.129 s |
| H100-80G | 13.086 s | 2.942 s |

Stage 1. Stage 2 on the A100 is 7.000 s, the same within noise. Medians over 100 steps (60 for the
overlap arms) after a discarded warmup, real Zenodo data, padded token median 242. The two rentals
agree to 1.8 % shipped and 3.7 % overlapped.

The H100 being slower than the A100 in the left column is not a typo, and it is the point. At the
shipped `num_workers: 0` the host data pipeline is serialised with compute and is 53 % of the step,
so the step tracks the host CPU rather than the card; the A100 sits idle for about seven tenths of
it. Overlap the loader and the H100 lands 2.4 % ahead of the A100, not a generation ahead. Device
time is the control: 2.049 s shipped against 2.078 s overlapped on the A100, unchanged while the
step moves 2.32x.

Full numbers, the device/host split and the caveats are in the state doc
`~/.coworker/state/train-e-gpu-baseline.md`.

## Running it

```sh
# on the rented box, as root
ROOT=/root/abb3 bash setup_gpu_box.sh          # deps, upstream at its pinned commit, Zenodo data
TAG=a100-1 SUITE=full bash run_sweep.sh        # the measurement sweep
```

`SUITE` is `full` (everything plus the runner-fidelity controls), `confirm` (stage 1 only, for the
second rental) or `modern` (both stages, no controls).

Then, back here:

```sh
python3 summarise.py results
```

## The pieces

| file | what it does |
|---|---|
| `setup_gpu_box.sh` | provisions the box: upstream at commit `18e4058`, its pins, the Zenodo data |
| `gpu_abb3_step.py` | the runner. Modes `real`, `dataonly`, `profile` |
| `run_sweep.sh` | the sweep, and the box/CPU/co-tenancy record |
| `clockwatch.sh` | samples SM clock, power and the compute-app list *during* every run |
| `summarise.py` | folds the JSON into the step table, the split and the comparison |
| `results/` | the measured JSON, clock traces and box records |

`gpu_abb3_step.py` is a runner beside upstream's `stages/train.py`, not a reimplementation:
their `LitABB3`, their `ABB3Loss`, their `ABB3DataModule`, their `RAdam`, their `params.yaml`.
Its docstring lists every place it departs from `train.py` and why; two of those departures
(`strategy="ddp"`, the DVCLive logger) can be switched back on with `--ddp` and `--dvclive` so the
cost of the departure is measured rather than assumed.

`--workers N` / `--pin-memory` override the shipped `num_workers: 0`. That is a departure from
what the paper ships, so anything measured with it is reported separately and never as the bar.
