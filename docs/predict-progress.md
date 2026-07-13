# `tt-bio predict` live progress — per-phase ticking fix

The live progress view shown during `tt-bio predict` skipped the trunk recycling
phase for several models, jumping straight to diffusion. This is a
display/progress-reporting bug only — the fold itself ran correctly; the live
view just wasn't getting per-phase / per-iteration updates.

## What was wrong

Every model reports progress to the live view by calling a `progress_fn(stage,
step, total)` hook that pushes a `stage` event into the display queue. The
expected sequence for a fold is

```
loading → msa → prep → trunk 0/N → trunk 1/N → … → trunk N/N → diffusion 0/M → … → confidence → saving
```

Three independent wiring bugs broke this:

- **boltz2** — the device-resident trunk recycling loop (`TrunkModule.forward`
  in `tt_bio/tenstorrent.py`) runs every recycling iteration on the device in
  one call and emitted a single `trunk` event at the end. The live view sat at
  `Trunk 0/N` then jumped to the final iteration, then to diffusion.
- **protenix-v2** — the worker wrapped the model's `progress_fn` and remapped
  `trunk` → `diffusion` (and dropped `step`/`total`), so trunk iterations were
  reported as a stepless `diffusion` and the trunk phase disappeared. The EDM
  sampler also emitted no per-step progress, so real diffusion never ticked
  either.
- **opendde** — `OpenDDE.fold` did not accept or forward a `progress_fn`, and
  the worker pre-emitted `diffusion` before calling `fold`, so the trunk phase
  (OpenDDE reuses the Protenix-v2 trunk) was never shown: `loading → diffusion`.

`esmfold2` / `esmfold2-fast` already ticked once per trunk iteration and once
per diffusion step; they were unchanged.

## The fix

One shared progress path: every model hands the same `progress_fn` hook into
its trunk loop and diffusion sampler, and the hook forwards `(stage, step,
total)` unchanged.

- `TrunkModule.forward` now takes `progress_fn` and emits
  `progress_fn("trunk", step=cyc, total=recycling_steps+1)` once per recycling
  iteration, mirroring the host fallback loop. `Boltz2.forward` passes its
  `progress_fn` into the resident trunk.
- `protenix.edm_sample` now takes `progress_fn` and emits
  `progress_fn("diffusion", step=k, total=n_step)` per step; `Protenix.fold`
  threads `progress_fn` through to it.
- `OpenDDE.fold` now takes `progress_fn` and forwards it into the Protenix-v2
  trunk and `edm_sample` it reuses.
- The protenix-v2 and opendde workers pass `report_progress` straight to the
  model as `progress_fn` (it has exactly that signature), removing the remapping
  wrapper; the opendde worker dropped its premature `diffusion` emit.

A device-free regression test (`tests/test_predict_progress.py`) locks the
contract: `report_progress` is a clean passthrough (trunk stays `trunk`, not
remapped), a trunk loop produces one `trunk` event per iteration, the display
advances the trunk bar monotonically with no `0 → diffusion` jump, and the
OpenDDE event sequence contains a trunk phase between `prep` and `diffusion`.

## Evidence (real runs, qb2 card 0)

All runs: `examples/prot_no_msa.yaml` (117-residue single-sequence protein),
`--single_sequence --sampling_steps 20 --seed 0 --debug --log`, default
`recycling_steps` (3 for boltz2/esmfold2, 10 for protenix-v2/opendde). The
`--debug --log` view prints one timestamped line per stage change per device.

**BEFORE** runs use `origin/main` @`d9b05db` (the branch base); **AFTER** runs
use this branch (`wk/tt-bio-predict-progress-fix`).

### boltz2

Before (`origin/main`):

```
10:51:05  [tt0] prot_no_msa
10:51:05  [tt0]   trunk 0/4
10:51:07  [tt0]   trunk 3/4        ← jumps 0 → 3, no 1/2
10:51:07  [tt0]   diffusion 0/20
10:51:08  [tt0]   diffusion 10/20
10:51:08  [tt0]   confidence
10:51:08  [tt0] ✓ prot_no_msa — 2.6s
```

After (this branch):

```
10:46:31  [tt0] prot_no_msa
10:46:31  [tt0]   msa
10:46:31  [tt0]   prep
10:46:32  [tt0]   trunk 0/4
10:46:57  [tt0]   trunk 1/4
10:46:57  [tt0]   trunk 2/4
10:46:58  [tt0]   trunk 3/4
10:46:58  [tt0]   diffusion 0/20
10:47:21  [tt0]   diffusion 10/20
10:47:21  [tt0]   confidence
10:47:21  [tt0]   saving
10:47:21  [tt0] ✓ prot_no_msa — 49.5s
```

Trunk now advances `0/4 → 1/4 → 2/4 → 3/4` one iteration at a time before
diffusion. (The first iteration is slow — cold kernel compile — then 1/2/3 run
warm; that is real per-iteration progress, not a jump.)

### protenix-v2

Before (`origin/main`):

```
10:51:17  [tt0] prot_no_msa
10:51:17  [tt0]   diffusion        ← stepless; trunk remapped to "diffusion", step/total dropped
10:51:23  [tt0]   saving
10:51:23  [tt0] ✓ prot_no_msa — 6.5s
```

After (this branch):

```
10:48:05  [tt0]   trunk 0/10
10:48:12  [tt0]   trunk 1/10
10:48:12  [tt0]   trunk 2/10
… (trunk 3/10 … 9/10) …
10:48:16  [tt0]   trunk 9/10
10:48:16  [tt0]   diffusion 0/20
10:48:22  [tt0]   diffusion 1/20
… (diffusion 2/20 … 18/20) …
10:48:23  [tt0]   diffusion 19/20
10:48:23  [tt0]   saving
10:48:23  [tt0] ✓ prot_no_msa — 26.6s
```

Trunk phase is back (`0/10 → … → 9/10`) and diffusion now ticks per step
(`0/20 → … → 19/20`).

### opendde

Before (`origin/main`):

```
10:50:38  [tt0] prot_no_msa
10:50:38  [tt0]   diffusion        ← no trunk phase at all (loading → diffusion)
10:50:48  [tt0]   saving
10:50:48  [tt0] ✓ prot_no_msa — 9.9s
```

After (this branch):

```
10:49:42  [tt0]   trunk 0/10
10:49:42  [tt0]   trunk 1/10
… (trunk 2/10 … 8/10) …
10:49:49  [tt0]   trunk 9/10
10:49:50  [tt0]   diffusion 0/20
10:49:51  [tt0]   diffusion 1/20
… (diffusion 2/20 … 19/20) …
10:49:51  [tt0]   saving
10:49:51  [tt0] ✓ prot_no_msa — 10.1s
```

OpenDDE now shows its trunk phase (it reuses the Protenix-v2 trunk) and ticks
diffusion per step, through the same shared path as Protenix-v2.

### esmfold2 (unchanged, verified)

Already ticked per iteration before this change; no code touched. Verified it
still advances smoothly:

```
10:52:15  [tt0]   trunk 0/4
10:52:15  [tt0]   trunk 1/4
10:52:15  [tt0]   trunk 2/4
10:52:15  [tt0]   trunk 3/4
10:52:18  [tt0]   diffusion 0/14
… (diffusion 1/14 … 12/14) …
10:52:24  [tt0]   diffusion 13/14
10:52:24  [tt0]   confidence
10:52:24  [tt0]   saving
10:52:24  [tt0] ✓ prot_no_msa — 37.1s
```

### esmfold2-fast (unchanged, verified)

Shares the identical trunk/diffusion progress path with esmfold2 (same
`_run_one_loop` trunk emitter, same `sample_structure` diffusion emitter); no
code touched. Verified:

```
10:52:58  [tt0]   trunk 0/4
10:52:59  [tt0]   trunk 1/4
10:52:59  [tt0]   trunk 2/4
10:52:59  [tt0]   trunk 3/4
10:52:59  [tt0]   diffusion 0/14
… (diffusion 1/14 … 13/14) …
10:53:00  [tt0]   confidence
10:53:00  [tt0]   saving
10:53:00  [tt0] ✓ prot_no_msa — 4.6s
```

## Regression test

`tests/test_predict_progress.py` (device-free, 4 tests) — run with the tt-bio
env:

```
PYTHONPATH=$PWD ~/tt-bio-dev/env/bin/python -m pytest tests/test_predict_progress.py -v
```

All four pass on this branch.
