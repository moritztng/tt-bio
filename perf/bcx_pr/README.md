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
    bf304d1530  2026-09-26T07:2xZ     moritztng         Pin design workers by the visibility variable's own names
    d00dd6f243  2026-09-26T12:0xZ     moritztng         Drop the memory rows nvidia-smi cannot measure instead of raising

`bf304d1` is `PIN_FIX.patch` pushed to the branch on 2026-09-26 07:29Z after 11.8 h of silence on
two comments that both offered it. One additive commit, no rebase, no force-push. Re-measured
against the branch the same pass: `fix_check_at_bf304d1.out`, control `68b853dde` 4 pass / 5 fail,
result `bf304d1` 9 pass / 0 fail, plus an unstubbed run on the real CPU backend where both
revisions give `selected_design_gpus() -> []` and `plan_design_workers({}, 300, 20) -> []`.
`PR19_COMMENT_3_2026-09-26.md` is the comment as posted
(https://github.com/PacesaLab/BindCraft2/pull/19#issuecomment-5844267891).

So the PR is now +69/-24 across 4 files, not the +290/-15 we opened, and the 215-line
`tests/test_design_workers.py` we added came out in `68b853d`. The rewrite replaces our env-var
enumeration with `jax.devices()` and derives the visibility variable from the device kind.
Everything under "Measured at HEAD" below was measured on `403ddcf8`, our commit, and does not
describe what is on the PR today.

`0001-Find-and-pin-design-workers-without-nvidia-smi.patch` is our commit. `PR_BODY.md` is the body
as posted. There is no Tenstorrent code, no `tt_bio` import and no `tenstorrent` extra in any of it.

## The branch did not fix the crash a third party reported in the function it rewrites

Issue #23, filed 2026-09-26T00:44:28Z by `c00jsw00` against `3e3563894`: a campaign on a DGX Spark
GB10 refuses with `could not convert string to float: '[N/A]'` before the first trajectory. The
board shares one memory pool with the host, so `nvidia-smi --query-gpu=memory.free,memory.total`
has no board figure to give: it prints `[N/A]` in both fields and **exits 0**, which is why
`check=True` does not trip and why `except (OSError, CalledProcessError)` does not catch the
`ValueError` that `float('[N/A]')` raises. It escapes `plan_design_workers` and `cli.py:211` turns
it into `campaign refused`.

**`bf304d1` carried that line unchanged from `main`, so this branch reproduced the report exactly.**
`gb10_repro.py` / `.out`, four trees, an `nvidia-smi` stub that prints what the reporter's does and
`design_devices()` stubbed, so no accelerator is opened anywhere:

    upstream/main 3e3563894                         RAISES ValueError
    upstream/main + the reporter's own file          {} and a single-process campaign
    PR #19 merged into main, before d00dd6f          RAISES ValueError
    PR #19 merged into main, after  d00dd6f          {'0': (119.0, 119.0)} and a 4-worker plan

`d00dd6f` drops a row whose memory does not parse rather than the whole reading, which leaves
`design_gpu_memory_gb` to fall through to `jax_device_memory_gb()` as it already does where
`nvidia-smi` is absent. A board that reports numbers is read exactly as before.

**The 119.0 GB in the last row is the stub's number, not a GB10's.** What the CUDA plugin reports
for a unified pool is untested here and we have no such box. If it reports a `bytes_limit` the
packing is memory-aware; if it reports nothing the result is the one worker the reporter's own
patch gives. Neither refuses the campaign, which is the whole claim.

`fix_check.py` grew case F for this and now runs 11 checks. `fix_check_gb10.out`, measured on the
**merge result** rather than the branch tip, because a branch-tip reading does not verify the merge:

    upstream/main 3e3563894                          4 pass /  7 fail
    this branch merged into it, before d00dd6f       9 pass /  2 fail
    the same merge, after                           11 pass /  0 fail

The 2 are the GB10 pair. `git merge origin/main` resolves clean at `e4f58043c`, and the four files
this branch touches appear in none of the open PRs #10, #16, #21 or #22.
`ISSUE23_COMMENT_2026-09-26.md` and `PR19_COMMENT_4_2026-09-26.md` are the two comments as posted
(issuecomment-5846138314 and -5846138434). `GB10_NA.patch` is the commit.

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
