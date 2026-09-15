# Re-gate of `TT_BIO_SDPA_FUSED_LARGE_S` default-ON on the 66-commit-newer main

Status: **the merge is done and the re-gate is running.** No verdict yet from this pass. The lever
itself is unchanged and still GO on the evidence banked in `../gate/REPORT.md` and
`../gate2/REPORT.md`; what this pass owed was a re-measurement on the tree that now ships, and that
is what `perf/ttx_a3/gate3/` collects.

## What the re-merge changed, and what it therefore owes

`origin/main` moved 66 commits (`cdd5fddf2` -> `408e93d6e`). The merge is `5aef1d601`, conflict-free,
and the branch's code diff against main is byte-for-byte the same five files as before: the flag
default, the `SDPA_FUSED_LARGE_S_STATS` counter, one README row, one docs section, one test
(+37/-15). Verified on the merged tree: it imports, `_SDPA_FUSED_LARGE_S` reads `True`, and the
counter is wired.

No commit in those 66 touches `SDPA_FUSED_LARGE_S` (`git log -S`, empty). But eight of them touch
`tt_bio/tenstorrent.py`, and they are the `roof-transition-l1-1024-overflow-fix` line that made
`TT_BIO_TRANSITION_L1_ROWS` a shipped default. That matters here for one specific reason:

**Two levers that each claim "bit-identical below the cap" have never been measured together.**
`TT_BIO_TRANSITION_L1_ROWS` sizes each transition's row block from the card's L1 budget and its
README row claims bit-identical output at 298, 512, 768 and 1024 residues. This lever claims the
path below 1024 is untouched byte for byte. Both are now on by default, and perturbations stack
sub-additively rather than independently, so the stack has to be approved as a stack. The gate3
driver therefore runs the below-cap neutrality sizes FIRST, off/on/off, one arm per process:

    neut768-{off1,on1,off2}     the one size below the cap whose on-arm has never completed
    neut1024-{off1,on1,off2}    the cap boundary, and the shape TRANSITION_L1_ROWS moves

768 aa is the real gap. gate2 got one off-arm fold at 47.196 s and then three timeouts, and the
reason to retry it now rather than record it as unmeasurable is that main has since root-caused
that class of 1024-aa stall to this box's watchdog, not to L1 and not to this lever.

## gate2's parity arm was never an accuracy result

Worth stating plainly because its `rc=1` reads like a 28-leg accuracy failure and is not one. Every
one of the 28 ERROR legs failed identically:

    DeviceInUseError: physical card 0 on tt-quietbox2 is in use by
    worker:ttx-a3-sdpa-ship-remerge (pid 10158) -- the same holder identity in a DIFFERENT
    process, so this is a real co-tenant, not a stale lease; waited 120s.

One leg leaked a live process still holding card 0's flock, and every leg behind it burned the
lease's 120 s timeout and died. 12 legs passed before the leak, 28 failed after it, and the whole
arm produced no evidence about anything. The lease message is accurate — pid 10158 really was alive
and really did hold the card — which is why the arm's exit code had to be read through 28
tracebacks before it meant anything. `gate_drive.sh` now logs the card's holders BEFORE the parity
arm starts, so a leak that predates the arm is a one-line refusal instead of an hour of timeouts.
A leak that happens mid-arm belongs to `full_parity_gate.py`'s leg reaping, which is shared code
and may not change while the arm resumes across it.

pid 10158 is dead and card 0 is free, so the arm reruns clean in gate3.

## The host, measured this pass rather than remembered

qb2 had been reset-free for 4 h 29 m at load ~1 when this pass started, which is why it was worth
committing the gate to it. It then went down 6 seconds into the first fold's card-0 open (fold
started 18:38:39Z, `client_loop: send disconnect: Broken pipe`, box back up 18:42Z), and came back
by itself — a watchdog reset, not the silent hang. That is consistent with QBROOT's closed verdict
that the root cause is a per-card PCIe link failure: a calm box is not a safe box, the card open is
the trigger, and the only defence available to a gate is the one already built here, which is to
make every arm resumable and let cron relaunch.

## Blocker carried forward, unchanged and still not this task's call

main's `docs/size_ladder_baseline.d/{boltz2,esmfold2}.json` p300c entries are stale against nine
already-merged, already-default-shipping levers, so the size-ladder arm is red for every branch
including main itself with this flag off. `tt-bio-sizeladder-p300c-refresh` owns it and is still
in flight: its own state file records boltz2's gated verdict as PASS and esmfold2 still missing the
1024 rung after six reset-killed attempts, with no `VERDICT-SIZELADDER-P300C` line written. Until
that lands and this branch re-merges main to pick up the refreshed baseline, the ladder arm cannot
go green here for reasons that have nothing to do with this lever.

## Reused, not rebuilt

`perf/ttx_a3/gate_drive.sh` and `resume_after_boot.sh` (the resumable driver and its cron relaunch,
both parameterized by `GATE_OUT` precisely so a re-merged tree starts on a fresh progress file),
`perf/ttx_a3/fold_parity_a3.py` for the one-arm-per-process neutrality folds, and the banked 1536 aa
numbers and below-cap digests, which this pass does not re-measure.
