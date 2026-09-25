# The BC2 device arm aborts at teardown on the merged tree, and does not at the pre-fix commit

qb2 card 1, p300c, `bcx_e2e_venv` interpreter, `PYTHONPATH=<tree>:/home/ttuser/bcx_e2e/bc2`.
Raw logs are beside this file under `out/gate_land_merge/` (gitignored, so they live on the
worktree only).

| tree | what | step | result | teardown |
|---|---|---|---|---|
| `2f19af43a` | merged, carries the TrunkPool race fix `1d1856090` | 442.0 s | assertions passed | **ABORTED** |
| `3997446c6` | the commit BEFORE the race fix | 418.7 s | 1 passed in 426.71 s | clean |

The abort:

    Fatal Python error: terminate called after throwing an instance of std::runtime_error
    "pthread_mutex_unlock failed for mutex CHIP_IN_USE_1_PCIe errno: 1

`errno 1` is EPERM: a thread released a UMD per-chip mutex it does not own. Both arms compute
identically (loss 11.6109, `calls {primal 0, taped 2, backward 1}`), so the difference is
confined to process teardown.

## What this does and does not establish

It does NOT settle that `1d1856090` causes it. Two limits, both live:

1. **n=1 per arm**, against a suspected *race* at teardown. A race is exactly what one clean run
   cannot rule out.
2. **The arms differ by two things.** The merged tree carries `1d1856090` *and* the 12-commit
   `origin/main` merge. That merge brings no `tt_bio/bindcraft2.py` change (it moves
   `tt_bio/size_limits.py` plus `perf/`/docs), which is why the fix is the live suspect -- an
   argument, not a measurement.

Closing it is ~4 runs at ~7 min: repeats on both arms, plus one at `1d1856090` unmerged.

## Two control rungs that tested nothing, kept so nobody re-runs them

- **A bare `ttnn.open_device` is not a teardown control.** It dies first in the open:
  `TT_FATAL @ tt_cluster.cpp:273 is_custom_fabric_mesh_graph_desc_path_specified()`. tt-bio sets
  up fabric config a raw ttnn open lacks, so the process never reaches teardown holding a device.
- **`UMD |` / "Opening user mode device driver" is not a did-it-open-a-card marker.** tt-bio
  suppresses UMD logging: the run that indisputably used the card (442 s step, real loss) has a
  zero count for both strings, identical to a run that never opened one. Use a **negative
  control** instead -- with `TT_VISIBLE_DEVICES=` the device tests *skip*, and the skip reason
  says so. That is how `tests/test_pair_row_blocks.py` was confirmed a real device run
  (8 skipped without a grant, 8 passed in 32 s with one, zero abort lines).
