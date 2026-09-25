# The BC2 device arm aborts at teardown on the merged tree, and does not at the pre-fix commit

qb2 card 1, p300c, `bcx_e2e_venv` interpreter, `PYTHONPATH=<tree>:/home/ttuser/bcx_e2e/bc2`.
Raw logs are beside this file under `out/gate_land_merge/` (gitignored, so they live on the
worktree only).

| tree | what | step | result | teardown |
|---|---|---|---|---|
| `2f19af43a` | merged, carries the TrunkPool race fix `1d1856090` | 442.0 s | test body ran; **pytest never printed a summary** | **ABORTED** |
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

---

# Root cause, established by a three-arm ladder

Each comparison moves one variable. Arms were verified distinct before launch by `RLock` presence
and the `tt_bio/size_limits.py` commit, not assumed:

| arm | tree | race fix `1d1856090` | carries `origin/main` merge |
|---|---|---|---|
| `merged`  | `198e373a5` | present | yes (`size_limits` at `2d7c367d6`) |
| `fixonly` | `1d1856090` | present | no  (`size_limits` at `9944dddde`) |
| `prefix`  | `3997446c6` | absent  | no  (`size_limits` at `9944dddde`) |

`fixonly` vs `prefix` isolates the fix; `merged` vs `fixonly` isolates the merge. Serial on qb2
card 1, one device context per process, run back-to-back by `teardown_chain.sh` so host load is
comparable.

### Results

| arm | step | rc | abort lines | verdict |
|---|---|---|---|---|
| `merged` (pass 1, 18:52Z) | 442.0 s | - | 2 | **ABORTED**, no pytest summary |
| `merged` (pass 2, 19:43Z) | 420.3 s | 134 | 2 | **ABORTED** |
| `fixonly` (19:36Z) | 407.9 s | 134 | 2 | **ABORTED** |
| `prefix` (pass 1, 19:2xZ) | 418.7 s | 0 | 0 | clean, `1 passed` in 426.71 s |
| `prefix` (pass 2, 19:51Z) | 419.6 s | 0 | 0 | clean, `1 passed` in 426.83 s |

`rc=134` is SIGABRT (128+6).

**The abort tracks the race fix, not the merge: 3 of 3 runs carrying it abort, 0 of 2 without it do.** Three aborts across two independent trees that
carry `1d1856090`, including one tree with no `origin/main` merge at all, against a clean run on
the tree without it. That retires both limits the previous pass recorded: the merge is exonerated
(`fixonly` has none and still aborts) and the abort is deterministic on the merged tree rather
than an intermittent race (n=2, both aborting).

Step time is unaffected -- 407.9 to 442.0 s across every arm, aborting or not -- so this is a
teardown defect, not a compute one.


## The mechanism, not just the correlation

`1d1856090` moves `_Trunk` construction out of `TrunkPool.use()` and into the lazy `current`
property. `current` is reached from `_trunk()`, `_primal`, `_taped` and `_backward`, and those are
`jax.pure_callback` targets -- `jax.pure_callback(self._backward, ...)` at `bindcraft2.py:519`.

So after the fix, the first `_Trunk` construction, and with it the device bring-up that takes UMD's
per-chip CHIP_IN_USE lock, happens inside a JAX callback on a thread JAX chooses. At interpreter
exit the release runs on the main thread. Unlocking a pthread mutex from a thread that does not own
it is EPERM, which is the reported `errno: 1`, and a C++ wrapper raising inside a destructor ends
the process in `terminate`.

**The commit's own docstring already names this hazard** -- "jax.pure_callback is not guaranteed to
run on the calling thread" -- where it argues against making `_current` thread-local. It then
routed the device open through that same callback.

Before the fix, `use()` built the trunk eagerly on the caller's thread, so the open and the close
agreed on a thread.

## The repair this points to

Keep `use()` deferred, so the compile thread still never builds a trunk. That half of `1d1856090`
is correct and fixes a real abort (`context_id ... is out of range` out of
`Device::init_command_queue_device_with_topology`). Add one warm-up on the folding thread: in
`TenstorrentAlphaFoldDesignModel.sequence_gradients` and `.predict` (`bindcraft2.py:952` and
`:945`), inside `with self._route(name)`, touch `self.pool.current` when the call is **not**
`compile_only`. BindCraft 2 passes `compile_only=True` from its compile thread and that kwarg
arrives through `**kwargs`, so the two threads are distinguishable exactly there.

That gets both properties at once: the compile thread never constructs a trunk, and the device is
opened and closed by the same thread.

**Unverified.** This is a specification derived from the ladder, not a measured result. It needs
its own arm before it goes near a merge.
