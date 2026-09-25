# The upstream CPU reference for BoltzGen designability

A number from the device is not a defect until the reference's own number at the same size is
in hand. This directory is that reference's identity: what it is, what was changed to make it
run without a GPU, and what was deliberately left alone.

## What it is

Upstream `boltzgen==0.3.2` from PyPI — the same version
`docs/implementation-parity-data/boltzgen.json` records for the committed RTX 3090 reference —
on `torch 2.14.0+cpu`, in `qb2:~/bgref-env`. Weights resolve to HuggingFace snapshot
`c1be29e1f82ffcc72264f64b993c43fb4e0d17f0`, again the snapshot that file records, so this
reference and the committed one differ in hardware and dtype and in nothing else.

It runs on qb2 (16 cores, 249 GiB) with `CUDA_VISIBLE_DEVICES` and `TT_VISIBLE_DEVICES` both
empty. **No Tenstorrent card is opened.**

## The input is the same input, checked rather than assumed

`fx/` on qb2 holds the four fixture files copied off whglx from the directories the device
jobs actually read, and the fixtures `perf/bhdesign/ladder.py` generates on this branch are
md5-identical to them:

    bg512.yaml   41e05c48c9c29fed69439c4480bc58b8      bgt512.cif   a02e5edb3d4690ab610c6d5c0d736bc9
    bg1536.yaml  ef22632668f3ade572e37cd868392c31      bgt1536.cif  d896104c06f4e1632287bbcbc8d60ed1

## What was changed, and in which direction

`cpu_capability.patch` — upstream calls `torch.cuda.get_device_capability()` unconditionally
in `BinderDesignPipeline.__init__`, before the branch that is the only consumer of the value,
so a CPU-only torch raises `Torch not compiled with CUDA enabled` before any model is built.
The patch makes the probe conditional. With `--use_kernels false` the resulting value is the
same one the unpatched code would reach on any card, so this changes no arithmetic. tt-bio's
vendored copy dropped the probe entirely and hardcodes `use_kernels = False`.

`run.sh` overrides three Lightning settings per step, each recorded because each moves the
reference toward being MORE exact, not less:

| setting | shipped GPU config | here | why |
|---|---|---|---|
| `trainer.accelerator` | `gpu` | `cpu` | there is no card |
| `trainer.precision` | `bf16-mixed` | `32` | the reference is the arithmetic ground truth both bf16 sides approximate |
| `matmul_precision` | `high` (TF32 on an NVIDIA card) | `highest` | same reason |

## What was deliberately left alone

Everything the design itself depends on stays at the upstream default, because doing less of
the model's own work is a cheat rather than a saving: `sampling_steps 500`,
`recycling_steps 3`, `diffusion_samples 1`, `protocol protein-anything`. Those are the values
in the generated `config/design.yaml`, not values passed in.

## Running it

    bash run.sh 512 8        # size tag, host threads
    bash run.sh 1536 8

Score the result with the same harvest the device side uses — the metric is not
re-implemented on either side:

    python3 -c "from perf.mgxscale.job import designability; \
                print(designability(pathlib.Path('~/bgref-work/out512').expanduser()))"
