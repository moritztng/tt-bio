# qb2-768-1024-clean-ab — pass notes

## Live-state correction to the brief (2026-08-13)

The brief says the `_Q_PARALLEL` lever is release-gated and NOT merged on `wk/boltz2-sizes-perf`.
That is stale. It landed on main as commit `063f89db` ("triatt: ship the q-split ON at <=1024
padded tokens, gated off above"), which raised `_Q_SPLIT_MAX_S` from 768 to 1024 in
`tt_bio/triatt_sdpa.py`. The lever ships ON by default at both target sizes today. Merged by the
follow-on task `boltz2-qparallel-768-1024-land`, whose own OOM-stress and confirm runs
(`perf/b2sizes/qsplit_stress_1024_qb1.json`, `qsplit_confirm_768_qb1.json`) were again qb1 only:
`host: tt-quietbox`, ttnn 0.67.4, grid 13x10.

So every number behind the shipped default is qb1's 130-core grid. qb2 is 110 cores and ttnn
0.68.0, and the lever's entire mechanism is a per-core split (`q_per_core == 1` in
`fill_preconditions`), so grid size is exactly the variable that could invert it. This pass is a
post-merge validation on the second host, not a pre-merge screen. If the win does not reproduce on
qb2 the finding is a live regression on main, not an unmerged lever left on the shelf.

## Harness precondition, checked

The e3b0b95b class of fix is present. `set_arm` in `perf/other512/fold_ab_multi.py` clears
`T._SDPA_Q_CHUNK_OVER_L1` and `PM._PM_OVER_L1` per arm, so the process-global refusal memos no
longer let one arm rewrite a later arm's config. Reference arms still run first.

Arm meaning on current main: `on` pins `PM._Q_SPLIT = False`, i.e. main before `063f89db`.
`qsplit` pins it True, i.e. main's shipped default at these sizes.

## Run

`perf/b2sizes/run_qb2_ab.sh`, under benchlock, qb2 card 3, `TT_VISIBLE_DEVICES=3`.
1024 first (`on,on,on,qsplit,qsplit`), then 768 (`on,on,on,qsplit`).
Started 13:31:11Z after waiting 24 min behind two other benchlock holders.
Output: `perf/b2sizes/qsplit_ab_1024_qb2.json`, `qsplit_ab_768_qb2.json`, log `qb2_ab.log`.

If this pass was cut off mid-run: the job is detached (setsid+nohup) and rooted in this worktree.
Read the two JSONs, then append the dated section to
`/home/moritz/.coworker/state/boltz2-sizes-perf.md` containing the literal string
`qb2-768-1024-clean-ab`.

## Attempt 1 hung the device, and it hung on the REFERENCE arm (2026-08-13 13:31-14:19Z)

`run_qb2_ab.sh` attempt 1 acquired benchlock at 13:31:11Z, compiled every program by 13:31:47Z
(`generated/inspector/*.yaml` stop growing at that timestamp, so nothing was still JIT-ing), and
then made no further progress for **33 minutes** until it was killed at 14:04Z. Evidence it was a
device hang and not a slow fold:

- `py-spy record -d 20` on pid 87647: **1999 of 1999 samples on one identical 14-frame stack**, leaf
  at `fold_ab_multi.py:136`, which is the *first* `ttnn.synchronize_device` in `timed_call` (the
  drain before the clock starts), not the timed call itself.
- The stack places it in the MSA stack, not the trunk: `_iteration` (tenstorrent.py:5761, the
  `self.msa(...)` call) -> `MSA.__call__` block loop (4386) -> `MSALayer.__call__` at
  `z = self.pairformer_layer(...)` (4319). So the ops being drained are the MSA layers own

## Attempt 1 hung the device, and it hung on the REFERENCE arm (2026-08-13 13:31-14:19Z)

`run_qb2_ab.sh` attempt 1 acquired benchlock at 13:31:11Z, compiled every program by 13:31:47Z
(`generated/inspector/*.yaml` stop growing at that timestamp, so nothing was still JIT-ing), and
then made no further progress for **33 minutes** until it was killed at 14:04Z. Evidence it was a
device hang and not a slow fold:

- `py-spy record -d 20` on pid 87647: **1999 of 1999 samples on one identical 14-frame stack**, leaf
  at `fold_ab_multi.py:136`, which is the *first* `ttnn.synchronize_device` in `timed_call` (the
  drain before the clock starts), not the timed call itself.
- The stack places it in the MSA stack, not the trunk: `_iteration` (tenstorrent.py:5761, the
  `self.msa(...)` call) then `MSA.__call__` block loop (4386) then `MSALayer.__call__` at
  `z = self.pairformer_layer(...)` (4319). So the ops being drained are the MSA layer's own
  (pair_weighted_averaging / msa_transition / outer_product_mean, or the chunked-rows path).
- 28+ min of *user* CPU (`/proc/87647/stat` utime 170667 ticks) with zero file writes anywhere:
  `synchronize_device` busy-polls, so 100 % CPU is the signature of this hang, not of progress.
- Two caught `TT_THROW` CB overflows at 13:31:23 (3394048 B and 2343424 B on core range
  (0,0)-(10,9)) precede it. Those are the harness's normal L1 refusals, absorbed, not the hang.

**The arm matters: this was NOT the q-split.** `main()` calls `set_arm("on")` before `build_fold`,
so the cold fold runs with `PM._Q_SPLIT = False`, i.e. main *before* `063f89db`. The config that
hung is the shipped baseline at 1024 aa on a 110-core grid, and the lever under test had not been
enabled yet.

**Card 3 was not left dirty by another job.** qb2 booted 11:55:40Z and
`state/leases/tt-quietbox2-card3.json` shows exactly one lease since then, this task's own at
13:31:13Z. The signature is not `qb2-dispatch-deadlock-reset-proof-needs-reboot` either: that one
hangs at ~3 s CPU inside the device-open dispatch probe, this one ran 35 s of real work and wrote
5.3 MB of `mesh_workloads_log.yaml` first.

Recovery: `kill -INT` did nothing (the hang is inside a native poll), then `kill -9 87647` and
`tt-smi -r 3`. UMD chip ids on qb2 are 1:1 with `/dev/tenstorrent/N` (verified with `tt-smi -ls`),
so `-r 3` is the same card as `TT_VISIBLE_DEVICES=3`. Runner chain 87645/79214/79212 killed by
explicit pid so the 768 leg could not start on a dirty card.

Attempt 2 launched 14:22Z, same arms, with a `timeout` per leg (2400 s for 1024, 1800 s for 768) so
a repeat hang cannot sit on benchlock unbounded. It queued behind `openfold3-to-3x-perdollar`, which
took benchlock at ~14:04Z for its own 1024 aa openfold3 leg.

**Open question this pass could not settle:** whether the hang reproduces. If attempt 2's cold 1024
`on` fold completes, attempt 1 was a one-off on a freshly booted card. If it hangs again, boltz-2 at
1024 aa does not run on qb2 at all right now, and that is a live main-branch finding independent of
the q-split.
