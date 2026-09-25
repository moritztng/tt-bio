# BindCraft 2 PR 1 — vendor-neutral design-worker enumeration and pinning

**Posted: https://github.com/PacesaLab/BindCraft2/pull/19**, opened 2026-09-24 as
`moritztng:vendor-neutral-design-workers` against upstream `main` at
`301efdd1937fb963cc40a5b0ecc1bc2f9b2b2d15` (2026-09-24T17:16:08Z).

## The PR is no longer the commit we posted

`martinpacesa` closed it unmerged at 2026-09-24T22:57:24Z. `LeonardoTredese` reopened it at
2026-09-25T09:02:47Z, **pushed two commits of his own onto our branch**, and requested review
from `ErikMaeots` at 2026-09-25T15:04:09Z.

    403ddcf888  2026-09-24T20:42:21Z  moritztng         Find and pin design workers without nvidia-smi
    44a0dd34b3  2026-09-25T12:12:37Z  LeonardoTredese   unified gpu determination mechanism
    68b853ddec  2026-09-25T15:00:48Z  LeonardoTredese   Updated docs, removed stale tests, fixed bindcraft.py imports

So the PR is now +49/-23 across 4 files, not the +290/-15 we opened, and the 215-line
`tests/test_design_workers.py` we added came out in `68b853d`. The rewrite replaces our env-var
enumeration with `jax.devices()` and derives the visibility variable from the device kind.
Everything under "Measured at HEAD" below was measured on `403ddcf8`, our commit, and does not
describe what is on the PR today.

`0001-Find-and-pin-design-workers-without-nvidia-smi.patch` is our commit. `PR_BODY.md` is the body
as posted. There is no Tenstorrent code, no `tt_bio` import and no `tenstorrent` extra in any of it.

## Two defects in the rewrite, reported 2026-09-25T16:18:46Z

`PR19_COMMENT_2026-09-25.md` is the comment as posted
(https://github.com/PacesaLab/BindCraft2/pull/19#issuecomment-5835685486).
`maintainer_revision_repro.py` and `.out` are the reproductions, pc, jax 0.11.2, python 3.12, CPU
backend, no accelerator.

1. **`design_visibility_variable()` has no `None` return path**, so the `is None` guard at
   `design_workers.py:144` and the `or 'CUDA_VISIBLE_DEVICES'` fallback at `:225` are both dead. On
   a platform that is neither cuda nor rocm the campaign raises where the guard's own comment says
   it should fall back to one worker.
2. **Device ids are renumbered under a partial visibility variable.** At `CUDA_VISIBLE_DEVICES=2,3`
   the base returns `['2','3']` and the PR head returns `['0','1']`, and `launch_design_workers`
   writes those back, so the workers are pinned to two cards the job was not given.
   `design_gpu_memory_gb()` prefers `nvidia_smi_memory_gb()`, whose keys are physical, and looks
   them up by jax id, so packing reads the wrong card's free memory.

The id half is argued, not measured on NVIDIA hardware: what was measured here is that jax ids are
a dense enumeration of the devices a process can see. The comment says so and gives the one-line
check.

A fix for both was tested here before it was suggested: keep the variable's own values when it is
set, fall back to `jax.devices()` only when it is unset. That restores `['2','3']` under a subset
and still enumerates all four cards on a rocm box with no variable set.

## What changed between the pin and HEAD

Built first at pin `7a2dfdb8a2`, rebased onto `301efdd` for the PR. Three differences mattered.

- **oneAPI is gone.** `52df4e591d` ("removed intelgpu support (currently experimental for jax)")
  dropped the `oneapi` extra from `pyproject.toml` and `oneapi` from `install.sh`. The patch's
  `ZE_AFFINITY_MASK` handling and its two `oneapi`/`xpu` test rows came out, because a PR that
  re-adds support the maintainers just withdrew argues with them for nothing. 22 tests became 20.
  `selfcheck.py:6 ACCELERATOR_MODULES` still lists `oneapi`, which is harmless here: the plugin
  lookup intersects with the platforms we pin and that set no longer contains it.
- **`docs/installation.md` moved to `docs/source/installation.md`** in the Sphinx move (`843009829e`,
  PR #1) and the paragraph the patch edits was rewritten there. Resolved by hand against the new
  text rather than by `git am --3way`, which left a conflict.
- **`design_workers.py` itself is untouched since the pin**, so the code the patch changes is the
  code upstream is running.

## Measured at HEAD

| what | number |
| --- | --- |
| Tests passing, patched, fresh clone at `301efdd` | `20 passed in 2.15s` |
| Same tests with `design_workers.py` reverted to `301efdd` | `11 failed, 9 passed in 1.04s` |
| Diff | 290 insertions, 15 deletions, 4 files |
| Environment | jax 0.11.2, python 3.12.12, CPU backend, `[CpuDevice(id=0)]`, pc |

The 9 that pass at both revisions are the ones holding the CUDA path fixed.

The ROCm reproduction in the body was run here, on a box with no accelerator, because
`HIP_VISIBLE_DEVICES` is read before any device is opened: at `301efdd`,
`HIP_VISIBLE_DEVICES=0,1,2,3` gives `visible_design_gpus() == []` and a plan of 0 workers; with the
commit it gives `['0','1','2','3']` and a plan of 4. The earlier draft of this body carried a
console transcript from a 4-card ROCm machine we do not have. It was replaced with the run above.

Re-verified at HEAD rather than carried over: `JAX_ROCM_VISIBLE_DEVICES=3` leaves
`jax_rocm_visible_devices` at `all` on jax 0.11.2 while `JAX_ENABLE_X64=1` takes in the same
interpreter, which is why a worker subprocess is pinned with the runtime's own variable.

## Reproducing

```bash
git clone https://github.com/PacesaLab/BindCraft2.git && cd BindCraft2
git checkout 301efdd1937fb963cc40a5b0ecc1bc2f9b2b2d15
git am --3way /path/to/0001-Find-and-pin-design-workers-without-nvidia-smi.patch
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest tests/test_design_workers.py -q          # 20 passed
git checkout 301efdd -- bindcraft/design_workers.py
.venv/bin/python -m pytest tests/test_design_workers.py -q          # 11 failed, 9 passed
```

## Deliberately excluded

- **A Tenstorrent backend**, and the predictor-factory seam it needs. Those wait for a maintainer
  response on this PR; see `state/bcx/UPSTREAM.md` for the three-seam analysis.
- **`cli.py:155 design_card_name()`**, which names the compile cache from `nvidia-smi`. A JAX
  equivalent would initialise the backend before `use_campaign_compile_cache()` sets
  `JAX_COMPILATION_CACHE_DIR`, losing the persistent cache. Non-NVIDIA machines already fall back
  to `${TMPDIR}/bindcraft_xla_cache`, so this costs them a cache name, not a campaign.
- **Installer auto-detection.** `install.sh:16` takes `cuda13|cuda12|rocm` positionally already.
  The one CUDA-specific piece left is the `cuda13` fallback when no driver answers, which their own
  comment defends as the login-node case, and we cannot exercise the detection on either machine.
