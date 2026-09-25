# BindCraft 2 PR 1 — vendor-neutral design-worker enumeration and pinning

Patch: `0001-Find-and-pin-design-workers-without-nvidia-smi.patch`, one commit against
`PacesaLab/BindCraft2` at pin `7a2dfdb8a285232a6f881899fe135c6dc48679f1`.

Nothing here has been posted. Opening the PR is Moritz's to do (ask 10569).

There is no Tenstorrent code, no `tt_bio` import and no `tenstorrent` extra in the diff. It is a
fix for the `rocm` and `oneapi` extras the lab already ships.

---

## PR title

    Find and pin design workers without nvidia-smi

## PR body

> `pyproject.toml` ships `rocm` and `oneapi` extras, but a campaign on either of them designs on one
> device however many the node holds, because worker fan-out is the one part of the tree that still
> speaks CUDA.
>
> `visible_design_gpus()` reads `CUDA_VISIBLE_DEVICES` and otherwise shells out to `nvidia-smi`, so
> on a ROCm box it returns `[]`. `plan_design_workers()` then builds an empty plan,
> `dispatch_design_workers()` sees fewer than two workers and returns `None`, and the campaign runs
> a single process. `design_gpu_memory_gb()` returns `{}` for the same reason, which sends
> `design_workers_per_gpu()` down its `trajectory_only` branch and holds even one card to one
> worker. On a ROCm machine with four cards:
>
> ```
> $ python -c "import jax; print(len(jax.devices()))"
> 4
> $ python -c "from bindcraft.design_workers import visible_design_gpus; print(visible_design_gpus())"
> []
> ```
>
> **What the diff changes.** Enumeration reads `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or
> `ZE_AFFINITY_MASK`, then `nvidia-smi`, and falls back to `jax.devices()` with the host CPU left
> out. Device memory falls back to the device's PJRT `memory_stats()`. Each worker is pinned with
> the variable its own runtime reads, chosen from the device platform, with JAX's shared `gpu`
> platform name resolved by which `jax_plugins.*` module is installed.
>
> I used the runtime variables rather than JAX's own `jax_cuda_visible_devices` /
> `jax_rocm_visible_devices`, because those are `config.string_flag`s rather than environment
> states: `JAX_ROCM_VISIBLE_DEVICES=3` leaves the flag at `all`, so it cannot pin a worker
> subprocess. `jax.local_devices()[i]` with explicit placement was the other option and it would
> mean touching every `jit` in `af2.py`, where one process per device needs nothing but the right
> environment.
>
> **What it does not change.** Both `nvidia-smi` calls are tried first and are untouched, so on an
> NVIDIA machine the devices found, their order, their memory and the variable each worker is
> pinned with are exactly what they were, and no JAX backend is initialised any earlier than before
> (that last one matters: `bindcraft.py` prints its GPU note before `cli.design()` sets
> `JAX_COMPILATION_CACHE_DIR`, so reaching for JAX there would have cost the per-card compile
> cache). Nine of the tests describe behaviour the commit preserves and pass at either revision.
>
> Where a platform offers no way to pin a worker, the campaign says so and designs on one device
> rather than fanning out onto a device every worker would then share.
>
> `tests/test_design_workers.py` is new; the repo had no tests directory, and `setuptools`
> `packages.find` is already scoped to `bindcraft*` so it is not packaged.

---

## Reproducing the tests

No accelerator is needed. The tests use stub devices, and the two that describe an NVIDIA machine
drive the real `nvidia-smi` parsing through a fake `nvidia-smi` on `PATH`.

```bash
git clone https://github.com/PacesaLab/BindCraft2.git && cd BindCraft2
git checkout 7a2dfdb8a285232a6f881899fe135c6dc48679f1
git am --3way /path/to/0001-Find-and-pin-design-workers-without-nvidia-smi.patch

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest tests/test_design_workers.py -q
```

Expected: `22 passed`. Installing the base package with no extra gives CPU-only JAX, which is the
point: the suite runs on a machine with no accelerator at all.

### Showing the tests are red before the change

```bash
git checkout 7a2dfdb8a285232a6f881899fe135c6dc48679f1 -- bindcraft/design_workers.py
.venv/bin/python -m pytest tests/test_design_workers.py -q    # 13 failed, 9 passed
```

The 9 that still pass are the ones pinning behaviour the commit preserves:

| Test | What it holds fixed |
| --- | --- |
| `test_a_machine_with_no_accelerator_finds_no_device` | The host CPU is never mistaken for a device to design on |
| `test_the_host_cpu_is_never_a_design_device` | Same, against the real CPU-only JAX installed here |
| `test_cuda_visible_devices_is_taken_verbatim_and_jax_is_not_consulted` | `CUDA_VISIBLE_DEVICES=2,3` still gives `['2','3']` |
| `test_an_empty_cuda_visible_devices_still_finds_nothing` | Slurm's empty variable still means no card |
| `test_nvidia_smi_answers_before_jax_is_asked` | An NVIDIA box is enumerated by the same call as before |
| `test_nvidia_smi_memory_is_preferred_and_parsed_as_before` | Same MiB parsing, same GiB result |
| `test_a_plugin_that_reports_no_memory_is_left_unsized` | An unsized card still falls back to one worker |
| `test_a_cuda_worker_is_still_pinned_with_cuda_visible_devices` | A CUDA worker still gets `CUDA_VISIBLE_DEVICES` |
| `test_a_worker_still_carries_its_index_and_count` | `BINDCRAFT_WORKER_ID` / `_COUNT` unchanged |

## How far the CUDA evidence goes

We have no NVIDIA card. What is proved is that the `nvidia-smi` code path is tried first and is
byte-identical to the pin, that a fake `nvidia-smi` on `PATH` is parsed into the same indices and
the same GiB figures, and that a CUDA-platform worker is still handed `CUDA_VISIBLE_DEVICES`. What
is not proved is a real multi-GPU campaign, and the PR body should not claim otherwise.

## Deliberately excluded

- **A Tenstorrent backend.** That is a separate PR and it needs this one first.
- **`cli.py:155 design_card_name()`**, which names the compile cache from `nvidia-smi`. A JAX
  equivalent would initialise the backend before `use_campaign_compile_cache()` sets
  `JAX_COMPILATION_CACHE_DIR`, losing the persistent cache. Non-NVIDIA machines already fall back
  to `${TMPDIR}/bindcraft_xla_cache`, so this costs them a cache name, not a campaign.
- **An `--accelerator` installer flag.** `install.sh:16` already takes `cuda13|cuda12|rocm|oneapi`
  positionally and installs the matching extra. The brief for this row expected that to be missing;
  it is present at this pin. The one CUDA-specific piece left is auto-detection falling back to
  `cuda13` when there is no driver, which is a deliberate documented choice for login nodes.
  Detecting ROCm or oneAPI ahead of that fallback would be a reasonable follow-up, and we left it
  out because we cannot test the detection on either machine.
