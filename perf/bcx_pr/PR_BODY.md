`pyproject.toml` ships a `rocm` extra, but a campaign installed with it designs on one device
however many the node holds, because worker fan-out is the one part of the tree that still speaks
CUDA.

`visible_design_gpus()` reads `CUDA_VISIBLE_DEVICES` and otherwise shells out to `nvidia-smi`, so
on a ROCm box it returns `[]`. `plan_design_workers()` then builds an empty plan,
`dispatch_design_workers()` sees fewer than two workers and returns `None`, and the campaign runs a
single process. `design_gpu_memory_gb()` returns `{}` for the same reason, which sends
`design_workers_per_gpu()` down its `trajectory_only` branch and holds even one card to one worker.

`HIP_VISIBLE_DEVICES` is what a ROCm allocation sets, and it is the shape Slurm hands a job. On
this machine, with no accelerator at all, before and after the commit:

```
$ HIP_VISIBLE_DEVICES=0,1,2,3 python -c "from bindcraft.design_workers import visible_design_gpus, plan_design_workers; print(visible_design_gpus(), len(plan_design_workers({}, 300, 20)))"
[] 0          # 301efdd
['0', '1', '2', '3'] 4   # with this commit
```

**What the diff changes.** Enumeration reads `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`, then
`nvidia-smi`, and falls back to `jax.devices()` with the host CPU left out. Device memory falls
back to the device's PJRT `memory_stats()`. Each worker is pinned with the variable its own runtime
reads, chosen from the device platform, with JAX's shared `gpu` platform name resolved by which
`jax_plugins.*` module is installed.

I used the runtime variables rather than JAX's own `jax_cuda_visible_devices` /
`jax_rocm_visible_devices`, because those are `config.string_flag`s rather than environment states.
On jax 0.11.2, `JAX_ROCM_VISIBLE_DEVICES=3` leaves the flag at `all` while `JAX_ENABLE_X64=1` takes
in the same interpreter, so the flag cannot pin a worker subprocess. Explicit placement with
`jax.local_devices()[i]` was the other option and it would mean touching every `jit` in `af2.py`,
where one process per device needs nothing but the right environment.

**What it does not change.** Both `nvidia-smi` calls are tried first and their bodies are untouched,
so on an NVIDIA machine the devices found, their order, their memory and the variable each worker is
pinned with are exactly what they were, and no JAX backend is initialised any earlier than before.
That last one matters: `bindcraft.py:63` prints its GPU note through `visible_design_gpus()` before
`cli.design()` reaches `use_campaign_compile_cache()`, so reaching for JAX there would initialise
the backend before `JAX_COMPILATION_CACHE_DIR` is set and quietly cost the per-card compile cache.
Nine of the tests describe behaviour this commit preserves and pass at either revision.

Where a platform offers no way to pin a worker, the campaign says so and designs on one device
rather than fanning out onto a device every worker would then share.

`tests/test_design_workers.py` is new; the repo had no tests directory, and `setuptools`
`packages.find` is already scoped to `bindcraft*`, so it is not packaged. No accelerator is needed
to run it: the devices are stubs, and the two tests that describe an NVIDIA machine drive the real
`nvidia-smi` parsing through a fake `nvidia-smi` on `PATH`.

```
$ pytest tests/test_design_workers.py -q
20 passed

$ git checkout 301efdd -- bindcraft/design_workers.py && pytest tests/test_design_workers.py -q
11 failed, 9 passed
```

**Where my evidence stops.** We have no NVIDIA card here, so what is proved about the CUDA path is
that it is reached first, that its code is byte-identical to `301efdd`, that a fake `nvidia-smi` on
`PATH` parses to the same indices and the same GiB pair, and that a CUDA-platform worker is still
handed `CUDA_VISIBLE_DEVICES`. A real multi-GPU NVIDIA campaign is not something I can run, and I
am not claiming it.

I also left out `cli.py:155 design_card_name()`, which names the compile cache from `nvidia-smi`, for
the initialisation-order reason above: non-NVIDIA machines already fall back to
`${TMPDIR}/bindcraft_xla_cache`, so it costs them a cache name rather than a campaign. And I left
the installer's auto-detection alone, since it already takes `cuda13|cuda12|rocm` positionally and
the `cuda13` fallback on a driverless login node is a documented choice; detecting ROCm ahead of
that fallback would be a reasonable follow-up, but I cannot exercise the detection here.
