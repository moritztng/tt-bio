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
