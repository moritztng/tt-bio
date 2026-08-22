# OpenBind-0 GPU reference

The H200 numbers the tt-bio OpenBind-0 port is scored against. `gpu_reference.json` is the
machine-readable record; read `caveats` in it before quoting any row.

The bar is `tt_target_device_s = 4 x h200_device_s` at matched diffusion-sample count. Device
time, not wall clock: the port pays the same host featurisation on its own CPU, so host time is
common cost to both arms.

Three arms:

| arm | what |
|---|---|
| `ob` | OpenBind-0, openfold3 0.5.0 + `of3-ob-2025-06-30-174k.pt` |
| `p2` | OpenFold3 preview2, openfold3 0.4.5 + `of3-p2-155k.pt`, what tt-bio pins today |
| `confirm-ob` | the `ob` arm again on a second, independently rented H200 |

`ob` and `p2` ran back to back on one card in one session, so the OB0 delta is priced with the
hardware held fixed. Upstream made the two checkpoints mutually exclusive, so that delta is
weights *and* code version and cannot be separated.

Only the 1024 aa rung reproduces across boxes (+1.2%); 128-512 aa move 10-17% because the
200-step diffusion rollout is launch-bound and picks up the rented host's CPU. Quote 1024 aa as
the bar.

## Files

- `make_inputs.py` — regenerates `inputs/`. Each spec is emitted as a neutral `.spec.json`, the
  OpenFold3 query set (`.of3.json`) and the tt-bio YAML (`.tt.yaml`), with sha256 in
  `inputs/SHA256SUMS`. If the TT side folds something not in that list the comparison is void.
- `gpu_ob_setup.sh` — stages a rented box: system libs, both checkpoints, both venvs, and the
  cuEquivariance wheel fix (upstream's `[cuequivariance]` extra installs cu12 ops against a CUDA
  13 torch and does not put `libcue_ops.so` on the loader path).
- `gpu_ob_run.py` — one cell: 1 cold + 3 warm folds in one process, kernel paths counted, device
  time split from host time, card exclusivity and power recorded.
- `gpu_ob_sweep.sh` — drives all cells for one arm.
- `make_reference.py` — folds `results/` into `gpu_reference.json`.
- `results/` — raw per-cell reports, setup and sweep logs, and the 0.4.5 -> 0.5.0 package diff.

## Reproducing

```bash
# on the rented box
bash gpu_ob_setup.sh base ckpt ob cueqfix p2
bash gpu_ob_sweep.sh ob && bash gpu_ob_sweep.sh p2
# back here
python make_reference.py --results perf/openbind/results --out perf/openbind/gpu_reference.json
```

Full write-up, including the OB0-vs-preview2 attribution and the silent 128-token kernel
fallback: `~/.coworker/state/openbind-gpu-reference.md`.
